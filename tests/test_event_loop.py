from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from jarvis.runtime.event_loop import QueueEventLoop, SignalBridge


def run_in_background(fn) -> threading.Thread:  # type: ignore[no-untyped-def]
    thread = threading.Thread(target=fn, daemon=True)
    thread.start()
    return thread


def test_call_soon_runs_in_order_on_loop_thread() -> None:
    loop = QueueEventLoop()
    seen: list[tuple[int, bool]] = []
    for i in range(3):
        loop.call_soon(
            lambda i=i: seen.append((i, threading.current_thread() is threading.main_thread()))
        )
    loop.call_soon(loop.stop)
    loop.run()
    assert seen == [(0, True), (1, True), (2, True)]


def test_call_and_wait_returns_value_from_loop_thread() -> None:
    loop = QueueEventLoop()
    results: list[object] = []

    def worker() -> None:
        results.append(loop.call_and_wait(lambda: threading.current_thread().name, default="x"))
        loop.stop()

    run_in_background(worker)
    loop.run()
    assert results == [threading.main_thread().name]


def test_call_and_wait_propagates_errors() -> None:
    loop = QueueEventLoop()
    errors: list[BaseException] = []

    def boom() -> None:
        raise ValueError("bad")

    def worker() -> None:
        try:
            loop.call_and_wait(boom, default=None)
        except ValueError as exc:
            errors.append(exc)
        loop.stop()

    run_in_background(worker)
    loop.run()
    assert len(errors) == 1


def test_stop_releases_waiting_workers_with_default() -> None:
    loop = QueueEventLoop()
    got: list[object] = []
    started = threading.Event()

    def slow_dialog() -> str:
        loop.stop()  # e.g. SIGTERM arrives while a dialog is open
        return "approved"

    def worker() -> None:
        started.set()
        got.append(loop.call_and_wait(slow_dialog, default="denied"))

    thread = run_in_background(worker)
    started.wait()
    loop.run()
    thread.join(timeout=2)
    assert got == ["denied"]  # a stopping loop never reports approval


def test_call_and_wait_after_stop_returns_default_immediately() -> None:
    loop = QueueEventLoop()
    loop.stop()
    assert loop.call_and_wait(lambda: "ran", default="default") == "default"


def test_failing_callback_does_not_kill_the_loop() -> None:
    loop = QueueEventLoop()
    seen: list[str] = []
    loop.call_soon(lambda: 1 / 0)
    loop.call_soon(lambda: seen.append("still running"))
    loop.call_soon(loop.stop)
    loop.run()
    assert seen == ["still running"]


def test_signal_bridge_delivers_on_loop_thread() -> None:
    loop = QueueEventLoop()
    received: list[tuple[int, bool]] = []

    def handler(signum: int) -> None:
        received.append((signum, threading.current_thread() is threading.main_thread()))
        loop.stop()

    bridge = SignalBridge(loop, handler)
    bridge.install([signal.SIGUSR1])
    try:
        threading.Timer(0.05, lambda: os.kill(os.getpid(), signal.SIGUSR1)).start()
        started = time.monotonic()
        loop.run()
        assert time.monotonic() - started < 2
    finally:
        bridge.uninstall()
    assert received == [(signal.SIGUSR1, True)]
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL


def test_signal_bridge_ignores_other_signals_and_uninstalls_cleanly() -> None:
    loop = QueueEventLoop()
    bridge = SignalBridge(loop, lambda s: pytest.fail("unexpected signal"))
    bridge.install([signal.SIGUSR2])
    bridge.uninstall()
    assert signal.getsignal(signal.SIGUSR2) is signal.SIG_DFL
