"""Main-thread event loops and signal delivery for the background service.

The service's main thread runs an :class:`EventLoop`: AppKit's run loop on
macOS (needed for global hotkeys and dialogs), or the portable
:class:`QueueEventLoop` in headless mode. Other threads never touch the UI
directly. They hand work to the main thread with :meth:`EventLoop.call_soon`
or :meth:`EventLoop.call_and_wait`.

Signals (SIGTERM from launchd, SIGINT from a terminal) arrive through
:class:`SignalBridge`. Python's C-level handler writes the signal number to a
pipe (``signal.set_wakeup_fd``), and a blocking reader thread forwards it to
the loop with ``call_soon``. This works while the loop is inside native code or
showing a modal dialog, and involves no polling.
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class EventLoop(ABC):
    """A loop owned by the main thread. ``call_soon`` and ``stop`` are thread-safe."""

    def __init__(self) -> None:
        self._waiters: set[threading.Event] = set()
        self._waiters_lock = threading.Lock()
        self._stopping = threading.Event()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    @abstractmethod
    def run(self) -> None:
        """Run until :meth:`stop` (call on the main thread)."""

    @abstractmethod
    def call_soon(self, fn: Callable[[], Any]) -> None:
        """Schedule ``fn`` on the loop thread. Safe from any thread."""

    def stop(self) -> None:
        """Ask the loop to finish. Pending :meth:`call_and_wait` callers get their default."""
        self._stopping.set()
        with self._waiters_lock:
            waiters = list(self._waiters)
        for waiter in waiters:
            waiter.set()
        self._wake_for_stop()

    @abstractmethod
    def _wake_for_stop(self) -> None:
        """Make :meth:`run` return promptly after :meth:`stop`."""

    def call_and_wait(self, fn: Callable[[], T], default: T) -> T:
        """Run ``fn`` on the loop thread and return its result (call from a worker thread).

        Returns ``default`` if the loop stops before ``fn`` completes, so a worker
        waiting for, say, a confirmation dialog can never hang during shutdown.
        """
        if self.stopping:
            return default
        done = threading.Event()
        box: dict[str, Any] = {}

        def wrapper() -> None:
            try:
                if not self.stopping:
                    value = fn()
                    # A result produced while stopping (e.g. a dialog answered as
                    # Jarvis shuts down) is discarded: stopping never approves.
                    if not self.stopping:
                        box["value"] = value
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        with self._waiters_lock:
            self._waiters.add(done)
        try:
            self.call_soon(wrapper)
            done.wait()
        finally:
            with self._waiters_lock:
                self._waiters.discard(done)
        if "error" in box:
            raise box["error"]
        return box.get("value", default)  # type: ignore[no-any-return]


_STOP = object()


class QueueEventLoop(EventLoop):
    """Portable loop: the main thread executes callables from a queue."""

    def __init__(self) -> None:
        super().__init__()
        self._queue: queue.Queue[object] = queue.Queue()

    def run(self) -> None:
        while not self.stopping:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                item()  # type: ignore[operator]
            except Exception:
                log.exception("Event loop callback failed")

    def call_soon(self, fn: Callable[[], Any]) -> None:
        self._queue.put(fn)

    def _wake_for_stop(self) -> None:
        self._queue.put(_STOP)


class SignalBridge:
    """Deliver selected POSIX signals to an :class:`EventLoop` callback.

    Must be installed from the main thread. Dispatch happens on the loop thread.
    """

    def __init__(self, loop: EventLoop, handler: Callable[[int], None]) -> None:
        self._loop = loop
        self._handler = handler
        self._read_fd, self._write_fd = os.pipe()
        os.set_blocking(self._write_fd, False)
        self._previous: dict[int, Any] = {}
        self._previous_wakeup: int | None = None
        self._thread = threading.Thread(target=self._pump, name="jarvis-signals", daemon=True)

    def install(self, signums: Iterable[int]) -> None:
        for signum in signums:
            # The Python-level handler does nothing: the wakeup fd carries the
            # signal number to our reader thread. Installing it replaces the
            # default action (e.g. terminate on SIGTERM).
            self._previous[signum] = signal.signal(signum, lambda *_: None)
        self._previous_wakeup = signal.set_wakeup_fd(self._write_fd, warn_on_full_buffer=False)
        self._thread.start()

    def uninstall(self) -> None:
        signal.set_wakeup_fd(self._previous_wakeup if self._previous_wakeup is not None else -1)
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()
        os.close(self._write_fd)  # ends the reader thread
        if self._thread.ident is not None:
            self._thread.join(timeout=2)
        os.close(self._read_fd)

    def _pump(self) -> None:
        while True:
            try:
                data = os.read(self._read_fd, 64)
            except OSError:
                return
            if not data:
                return
            for signum in data:
                if signum in self._previous:
                    log.info("Received signal %d", signum)
                    self._loop.call_soon(self._dispatcher(signum))

    def _dispatcher(self, signum: int) -> Callable[[], None]:
        def dispatch() -> None:
            self._handler(signum)

        return dispatch
