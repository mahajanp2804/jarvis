"""Tell whatever rings timers that they changed.

Inside the service, that's its own ringer. A `jarvis` command in a terminal
sends the running service a ``timers-changed`` message instead. When no
service answers, ``changed()`` is False, so the reply can say the timer won't
ring until the service starts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from jarvis.runtime.control import ControlError, send_request


def _service_socket() -> Path:
    from jarvis.runtime.lifecycle import DaemonPaths  # lifecycle imports bootstrap

    return DaemonPaths.default().socket


class TimerNotifier:
    def __init__(self, socket_path: Callable[[], Path] | None = None) -> None:
        self._socket_path = socket_path  # None: the service's, looked up at call time
        self._local: Callable[[], bool] | None = None

    def attach(self, ringer_changed: Callable[[], bool] | None) -> None:
        """The service's own ringer (None to detach on shutdown)."""
        self._local = ringer_changed

    def changed(self) -> bool:
        if self._local is not None:
            return self._local()
        try:
            path = (self._socket_path or _service_socket)()
            reply = send_request(path, {"cmd": "timers-changed"}, timeout=2.0)
        except (ControlError, OSError, ValueError):
            return False
        return bool(reply.get("ok"))
