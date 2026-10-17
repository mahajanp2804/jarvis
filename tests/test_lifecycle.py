"""Service lifecycle: single instance, control requests, graceful shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from jarvis.ai.base import ChatMessage, Completion
from jarvis.ai.llama_cpp_backend import ModelLoadError
from jarvis.config import JarvisConfig
from jarvis.runtime.control import send_request
from jarvis.runtime.lifecycle import AlreadyRunningError, Daemon, DaemonPaths, InstanceLock
from tests.fakes import FakeLLM, FakePlatform, plan_json


@pytest.fixture
def short_dir() -> Iterator[Path]:
    # Unix socket paths must stay under 104 bytes on macOS.
    with tempfile.TemporaryDirectory(prefix="jv", dir="/tmp") as base:
        yield Path(base)


def test_long_home_falls_back_to_short_socket_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jarvis.runtime.lifecycle.runtime_dir", lambda: Path("/x" * 60))
    paths = DaemonPaths.default()
    assert len(str(paths.socket)) < 104
    assert paths.lock == Path("/x" * 60) / "daemon.lock"


def test_instance_lock_is_exclusive(short_dir: Path) -> None:
    first = InstanceLock(short_dir / "d.lock")
    first.acquire()
    with pytest.raises(AlreadyRunningError, match=f"pid {os.getpid()}"):
        InstanceLock(short_dir / "d.lock").acquire()
    first.release()
    second = InstanceLock(short_dir / "d.lock")
    second.acquire()  # free again after release
    second.release()


class BlockingLLM(FakeLLM):
    """Model whose generation blocks until aborted, like a long native call."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.aborted = threading.Event()

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        self.started.set()
        if not self.aborted.wait(30):
            raise AssertionError("never aborted")
        raise ModelLoadError("The model worker stopped.")

    def abort(self) -> None:
        self.aborted.set()


def run_daemon_in_main_thread(
    short_dir: Path, tmp_path: Path, llm: FakeLLM, script: Any
) -> tuple[Daemon, float]:
    """Run a headless daemon on the main thread; ``script(paths)`` drives it."""
    config = JarvisConfig.model_validate(
        {"general": {"log_dir": str(tmp_path / "logs")}, "ai": {"idle_unload_seconds": 0}}
    )
    platform = FakePlatform()
    daemon = Daemon(
        config,
        platform,
        headless=True,
        paths=DaemonPaths(lock=short_dir / "d.lock", socket=short_dir / "j.sock"),
    )
    # Inject the fake model into the runtime the daemon builds.
    import jarvis.runtime.lifecycle as lifecycle

    original = lifecycle.build_runtime

    def build(*args: Any, **kwargs: Any) -> Any:
        return original(*args, backend=llm, **kwargs)

    lifecycle.build_runtime = build  # type: ignore[assignment]
    stopped_at: list[float] = []

    def driver() -> None:
        try:
            assert daemon.wait_ready(5)
            script(daemon.paths)
        finally:
            stopped_at.append(time.monotonic())
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        threading.Thread(target=driver, daemon=True).start()
        assert daemon.run() == 0
    finally:
        lifecycle.build_runtime = original  # type: ignore[assignment]
    return daemon, time.monotonic() - stopped_at[0]


def test_headless_daemon_serves_and_stops_on_sigterm(short_dir: Path, tmp_path: Path) -> None:
    llm = FakeLLM(plan_json("respond", "Hello from the model."))
    replies: dict[str, Any] = {}

    def script(paths: DaemonPaths) -> None:
        replies["status"] = send_request(paths.socket, {"cmd": "status"})
        replies["rule"] = send_request(paths.socket, {"cmd": "ask", "text": "list apps"})
        replies["ai"] = send_request(paths.socket, {"cmd": "ask", "text": "say hello"})
        replies["activate"] = send_request(paths.socket, {"cmd": "activate"})
        replies["bad"] = send_request(paths.socket, {"cmd": "rm -rf"})

    daemon, shutdown_seconds = run_daemon_in_main_thread(short_dir, tmp_path, llm, script)
    status = replies["status"]
    assert status["ok"] and status["mode"] == "headless" and status["model_loaded"] is False
    assert "Safari" in replies["rule"]["text"]
    assert replies["ai"]["text"] == "Hello from the model."
    assert replies["activate"]["ok"] is False  # no dialogs when headless
    assert replies["bad"]["ok"] is False
    assert shutdown_seconds < 2
    assert not daemon.paths.socket.exists()
    assert not llm.loaded
    # lock released: a new instance could start
    lock = InstanceLock(daemon.paths.lock)
    lock.acquire()
    lock.release()


def test_shutdown_aborts_an_in_flight_model_request(short_dir: Path, tmp_path: Path) -> None:
    llm = BlockingLLM()
    outcome: dict[str, Any] = {}

    def script(paths: DaemonPaths) -> None:
        def ask() -> None:
            try:
                outcome["reply"] = send_request(
                    paths.socket, {"cmd": "ask", "text": "do something slow"}, timeout=30
                )
            except Exception as exc:
                outcome["error"] = exc

        threading.Thread(target=ask, daemon=True).start()
        assert llm.started.wait(5)  # the model is mid-generation now

    daemon, shutdown_seconds = run_daemon_in_main_thread(short_dir, tmp_path, llm, script)
    assert llm.aborted.is_set()
    assert shutdown_seconds < 3, "shutdown must not wait for the native call"
    assert not daemon.paths.socket.exists()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_daemon_process_exits_cleanly_on_signal(short_dir: Path, signum: int) -> None:
    """Real process: `jarvis daemon --headless`, isolated HOME, then a signal."""
    env = {
        **os.environ,
        "HOME": str(short_dir),
        "JARVIS_AI_ENABLED": "0",
        "JARVIS_DRY_RUN": "1",
    }
    env.pop("JARVIS_CONFIG", None)
    env.pop("JARVIS_LOG_DIR", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "jarvis", "daemon", "--headless"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sock = short_dir / "Library" / "Application Support" / "Jarvis" / "run" / "jarvis.sock"
    try:
        deadline = time.monotonic() + 15
        while not sock.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert sock.exists(), proc.stdout.read().decode() if proc.poll() is not None else ""  # type: ignore[union-attr]
        reply = send_request(sock, {"cmd": "ask", "text": "open safari"})
        assert reply["text"].startswith("[dry run] Would: Open Safari")
        started = time.monotonic()
        proc.send_signal(signum)
        assert proc.wait(timeout=10) == 0
        assert time.monotonic() - started < 2
        assert not sock.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
