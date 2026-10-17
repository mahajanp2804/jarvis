"""The background service: start-up, idle, and graceful shutdown.

Start-up:  single-instance lock -> runtime (model *not* loaded) -> control
           socket -> global shortcut -> signal bridge -> main-thread event loop.
Idle:      the main thread sleeps in the run loop; the control thread sleeps in
           accept(); the model is unloaded after its idle timeout. No polling.
Shutdown:  (SIGTERM from launchd/logout, SIGINT from a terminal)
           1. stop accepting control requests and unregister the shortcut
           2. set the cancel event, so pending tool calls stop before acting
              and verification waits end early
           3. abort the model request in flight (kills the model worker)
           4. wait (bounded) for the request thread to finish
           5. unload the model, close the audit log, release the lock
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import tempfile
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from jarvis import __version__
from jarvis.agent.orchestrator import Reply
from jarvis.bootstrap import Runtime, build_runtime
from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.paths import runtime_dir
from jarvis.platforms.base import PlatformAdapter
from jarvis.runtime.assistant import Assistant, DialogConfirmer
from jarvis.runtime.control import ControlServer
from jarvis.runtime.event_loop import EventLoop, SignalBridge
from jarvis.runtime.hotkeys import Hotkey, HotkeyError, HotkeyListener
from jarvis.ui.base import HeadlessUI, UserInterface

log = logging.getLogger(__name__)

SHUTDOWN_GRACE_SECONDS = 10.0
SOCKET_PATH_LIMIT = 100


class AlreadyRunningError(JarvisError):
    """Another Jarvis service holds the instance lock."""


@dataclass(frozen=True)
class DaemonPaths:
    lock: Path
    socket: Path

    @classmethod
    def default(cls) -> DaemonPaths:
        base = runtime_dir()
        socket_path = base / "jarvis.sock"
        if len(str(socket_path)) >= SOCKET_PATH_LIMIT:
            # Unix socket paths are limited to 104 bytes on macOS. Fall back to the
            # per-user temporary directory (private to this user on macOS).
            socket_path = Path(tempfile.gettempdir()) / f"jarvis-{os.getuid()}" / "jarvis.sock"
        return cls(lock=base / "daemon.lock", socket=socket_path)


class InstanceLock:
    """Exclusive ``flock`` on a file; released automatically if the process dies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fh = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            owner = fh.read().strip() or "unknown"
            fh.close()
            raise AlreadyRunningError(
                f"Jarvis is already running (pid {owner}).",
                hint="Use `jarvis service status`, or stop it with `jarvis service stop`.",
            ) from None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def reply_payload(reply: Reply) -> dict[str, Any]:
    return {
        "ok": True,
        "text": reply.text,
        "handled": reply.handled,
        "statuses": [r.status.value for r in reply.results],
    }


class Daemon:
    """Owns the long-running service process."""

    def __init__(
        self,
        config: JarvisConfig,
        platform: PlatformAdapter,
        *,
        headless: bool = False,
        paths: DaemonPaths | None = None,
    ) -> None:
        self.config = config
        self.platform = platform
        self.headless = headless
        self.paths = paths or DaemonPaths.default()
        self.started_at = time.time()
        self.loop: EventLoop | None = None
        self.runtime: Runtime | None = None
        self.assistant: Assistant | None = None
        self.hotkey: Hotkey | None = None
        self.hotkey_error: str | None = None
        self._cancel = threading.Event()
        self._ready = threading.Event()

    def wait_ready(self, timeout: float) -> bool:
        """For tests and embedding: wait until the service accepts requests."""
        return self._ready.wait(timeout)

    def run(self) -> int:
        """Run until a shutdown signal. Returns the process exit code."""
        lock = InstanceLock(self.paths.lock)
        lock.acquire()  # raises AlreadyRunningError
        control: ControlServer | None = None
        bridge: SignalBridge | None = None
        listener: HotkeyListener | None = None
        try:
            loop = self.platform.event_loop(headless=self.headless)
            ui: UserInterface = HeadlessUI() if self.headless else self.platform.user_interface()
            self.loop = loop
            runtime = build_runtime(
                self.config, self.platform, DialogConfirmer(loop, ui), cancel_event=self._cancel
            )
            self.runtime = runtime
            assistant = Assistant(
                runtime.orchestrator, ui, loop, show_success=self.config.service.show_success
            )
            self.assistant = assistant
            control = ControlServer(self.paths.socket, self.handle_control)
            bridge = SignalBridge(loop, self._on_signal)
            bridge.install([signal.SIGTERM, signal.SIGINT])
            control.start()
            if not self.headless:
                listener = self._register_hotkey(loop, assistant)
            log.info(
                "Jarvis %s service started (pid %d, %s, shortcut %s)",
                __version__,
                os.getpid(),
                "headless" if self.headless else "GUI",
                self.hotkey.symbol() if self.hotkey else "none",
            )
            self._ready.set()
            loop.run()
            log.info("Shutting down")
        finally:
            self._shutdown(control, listener, bridge, lock)
        return 0

    # -- requests ----------------------------------------------------------------------

    def handle_control(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle one control-socket request (runs on a control thread)."""
        loop, assistant, runtime = self.loop, self.assistant, self.runtime
        if loop is None or assistant is None or runtime is None or loop.stopping:
            return {"ok": False, "error": "Jarvis is starting up or shutting down."}
        command = request.get("cmd")
        if command == "status":
            models = runtime.models
            return {
                "ok": True,
                "version": __version__,
                "pid": os.getpid(),
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "mode": "headless" if self.headless else "gui",
                "hotkey": self.hotkey.symbol() if self.hotkey else None,
                "hotkey_error": self.hotkey_error,
                "dry_run": self.config.general.dry_run,
                "ai_enabled": models is not None,
                "model_loaded": bool(models and models.is_loaded),
                "model_loads": models.stats.loads if models else 0,
                "worker_pids": models.backend.process_ids() if models else [],
                "busy": assistant.busy,
                "requests": assistant.requests_handled,
            }
        if command == "activate":
            if self.headless:
                return {"ok": False, "error": "The service is running headless (no dialogs)."}
            loop.call_soon(assistant.activate)
            return {"ok": True}
        if command == "ask":
            text = request.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                return {"ok": False, "error": "'ask' needs non-empty text (max 2000 characters)."}
            future = assistant.submit(text, interactive=False)
            timeout = self.config.service.request_timeout_seconds
            try:
                reply = future.result(timeout=timeout)
            except FutureTimeout:
                return {"ok": False, "error": f"no reply within {timeout:g}s"}
            except Exception as exc:
                return {"ok": False, "error": f"request failed: {exc}"}
            return reply_payload(reply)
        return {"ok": False, "error": f"unknown command {command!r}"}

    # -- internals ---------------------------------------------------------------------

    def _register_hotkey(self, loop: EventLoop, assistant: Assistant) -> HotkeyListener | None:
        try:
            hotkey = Hotkey.parse(self.config.service.hotkey)
            listener = self.platform.hotkey_listener(loop)
            listener.register(hotkey, assistant.activate)
        except JarvisError as exc:
            # Keep running: `jarvis activate` and `jarvis ask` still work.
            self.hotkey_error = exc.message
            log.error("Global shortcut unavailable: %s %s", exc.message, exc.hint or "")
            return None
        self.hotkey = hotkey
        return listener

    def _on_signal(self, signum: int) -> None:
        log.info("Stopping on %s", signal.Signals(signum).name)
        if self.loop is not None:
            self.loop.stop()

    def _shutdown(
        self,
        control: ControlServer | None,
        listener: HotkeyListener | None,
        bridge: SignalBridge | None,
        lock: InstanceLock,
    ) -> None:
        started = time.monotonic()
        if self.loop is not None and not self.loop.stopping:
            self.loop.stop()  # releases any worker waiting on a dialog
        if control is not None:
            control.stop()
        if listener is not None:
            try:
                listener.unregister()
            except HotkeyError:
                log.warning("Could not unregister the shortcut", exc_info=True)
        self._cancel.set()
        runtime = self.runtime
        if runtime is not None and runtime.models is not None:
            runtime.models.abort()
        if self.assistant is not None:
            self.assistant.shutdown(timeout=SHUTDOWN_GRACE_SECONDS)
        if runtime is not None:
            runtime.close()
        if bridge is not None:
            bridge.uninstall()
        lock.release()
        log.info("Stopped in %.2fs", time.monotonic() - started)
