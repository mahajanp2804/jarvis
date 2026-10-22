from __future__ import annotations

import contextlib
import json
import socket
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from jarvis.runtime.control import MAX_MESSAGE, ControlError, ControlServer, send_request


@pytest.fixture
def sock_path() -> Iterator[Path]:
    # Unix socket paths must be short (104 bytes on macOS), so not pytest's tmp_path.
    with tempfile.TemporaryDirectory(prefix="jv", dir="/tmp") as base:
        yield Path(base) / "run" / "j.sock"


def echo_handler(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("cmd") == "boom":
        raise RuntimeError("secret detail")
    return {"ok": True, "echo": request}


def test_round_trip_and_permissions(sock_path: Path) -> None:
    server = ControlServer(sock_path, echo_handler)
    server.start()
    try:
        assert send_request(sock_path, {"cmd": "status"}) == {"ok": True, "echo": {"cmd": "status"}}
        assert stat.S_IMODE(sock_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(sock_path.parent.stat().st_mode) == 0o700
    finally:
        server.stop()
    assert not sock_path.exists()


def test_handler_errors_are_contained(sock_path: Path) -> None:
    server = ControlServer(sock_path, echo_handler)
    server.start()
    try:
        reply = send_request(sock_path, {"cmd": "boom"})
        assert reply["ok"] is False
        assert "secret detail" not in reply["error"]
        # the server keeps serving after a failure
        assert send_request(sock_path, {"cmd": "status"})["ok"] is True
    finally:
        server.stop()


def raw_request(path: Path, payload: bytes) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)
        sock.connect(str(path))
        # The server rejects an oversized message as soon as it passes the limit,
        # answers and closes, possibly before we finish sending or shut down our
        # side (ENOTCONN/EPIPE on macOS). Its answer is still there to read.
        with contextlib.suppress(OSError):
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := sock.recv(4096):
            data += chunk
    return json.loads(data)  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "payload",
    [b"not json\n", b"[1, 2]\n", b"x" * (MAX_MESSAGE + 10)],
)
def test_bad_requests_are_rejected(sock_path: Path, payload: bytes) -> None:
    server = ControlServer(sock_path, echo_handler)
    server.start()
    try:
        assert raw_request(sock_path, payload)["ok"] is False
    finally:
        server.stop()


def test_stale_socket_is_replaced(sock_path: Path) -> None:
    sock_path.parent.mkdir(parents=True)
    sock_path.write_text("stale")
    server = ControlServer(sock_path, echo_handler)
    server.start()
    try:
        assert send_request(sock_path, {"cmd": "x"})["ok"] is True
    finally:
        server.stop()


def test_client_errors(sock_path: Path) -> None:
    with pytest.raises(ControlError, match="isn't running") as info:
        send_request(sock_path, {"cmd": "status"})
    assert info.value.hint and "service start" in info.value.hint
    sock_path.parent.mkdir(parents=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(sock_path))  # bound but never listening -> refused
    with pytest.raises(ControlError, match="stale"):
        send_request(sock_path, {"cmd": "status"})


def test_path_length_is_checked() -> None:
    with pytest.raises(ControlError, match="too long"):
        ControlServer(Path("/nonexistent/" + "x" * 120 + ".sock"), echo_handler)
