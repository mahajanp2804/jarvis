"""Local control socket for the background service.

``jarvis service status``, ``jarvis activate`` and ``jarvis ask`` talk to the
running service over a Unix domain socket:

* The socket lives in a ``0700`` directory and is itself ``0600``. On macOS and
  Linux the peer's user id is also checked, so only the same user can connect.
* One JSON request per connection, one JSON reply, newline-terminated, at most
  64 KB.
* Requests can't bypass anything. ``ask`` goes through the same orchestrator,
  policy and confirmation dialogs as the shortcut.

The accept loop blocks in ``accept()`` (no polling). ``stop()`` closes the socket.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import struct
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jarvis.errors import JarvisError

log = logging.getLogger(__name__)

MAX_MESSAGE = 64 * 1024
_UNIX_PATH_MAX = 104  # sockaddr_un.sun_path on macOS (108 on Linux)

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class ControlError(JarvisError):
    """The control socket could not be used."""


def _peer_uid(conn: socket.socket) -> int | None:
    try:
        if sys.platform == "darwin":
            # LOCAL_PEERCRED (SOL_LOCAL=0, option 1) -> struct xucred {version, uid, ...}
            raw = conn.getsockopt(0, 0x001, 128)
            return int(struct.unpack_from("=II", raw)[1])
        if sys.platform.startswith("linux"):
            raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            return int(struct.unpack("3i", raw)[1])
    except OSError:
        return None
    return None


def _read_message(conn: socket.socket) -> bytes:
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_MESSAGE:
            raise ControlError("Control message too large.")
    return data


class ControlServer:
    """Serves control requests on a Unix socket, one thread per connection."""

    def __init__(self, path: Path, handler: Handler) -> None:
        if len(str(path)) >= _UNIX_PATH_MAX:
            raise ControlError(f"Socket path is too long for macOS: {path}")
        self.path = path
        self._handler = handler
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        if self.path.exists():
            # Only one service holds the instance lock, so a leftover socket is stale.
            self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)  # create the socket as 0600
        try:
            sock.bind(str(self.path))
        finally:
            os.umask(old_umask)
        sock.listen(8)
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="jarvis-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _serve(self) -> None:
        while True:
            sock = self._sock
            if sock is None:
                return
            try:
                conn, _ = sock.accept()
            except OSError:
                return  # socket closed by stop()
            threading.Thread(
                target=self._handle, args=(conn,), name="jarvis-control-conn", daemon=True
            ).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                uid = _peer_uid(conn)
                if uid is not None and uid != os.getuid():
                    log.warning("Rejected control connection from uid %s", uid)
                    return
                conn.settimeout(10)
                request = json.loads(_read_message(conn) or b"{}")
                if not isinstance(request, dict):
                    raise ControlError("Request must be a JSON object.")
                conn.settimeout(None)
                reply = self._handler(request)
            except (ControlError, ValueError, OSError) as exc:
                reply = {"ok": False, "error": str(exc)}
            except Exception:
                log.exception("Control request failed")
                reply = {"ok": False, "error": "internal error (see the Jarvis log)"}
            with contextlib.suppress(OSError):
                conn.sendall(json.dumps(reply).encode("utf-8") + b"\n")


def send_request(path: Path, request: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
    """Send one request to a running service and return its reply."""
    if not path.exists():
        raise ControlError(
            "Jarvis isn't running.",
            hint="Start it with `jarvis service start` (or `jarvis daemon` in a terminal).",
        )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(str(path))
            sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
            data = _read_message(sock)
        except (ConnectionRefusedError, FileNotFoundError) as exc:
            raise ControlError(
                "Jarvis isn't running (stale socket).",
                hint="Start it with `jarvis service start`.",
            ) from exc
        except TimeoutError as exc:
            raise ControlError(f"Jarvis did not answer within {timeout:g}s.") from exc
    if not data:
        raise ControlError("Jarvis closed the connection without answering.")
    reply: dict[str, Any] = json.loads(data)
    return reply
