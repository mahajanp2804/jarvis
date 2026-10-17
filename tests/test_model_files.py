from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jarvis.ai.model_files import (
    ModelFileError,
    check_model_file,
    download_model,
    sha256_file,
)
from jarvis.ai.registry import DEFAULT_MODEL, ModelSpec

PAYLOAD = b"GGUF" + bytes(range(256)) * 40  # small stand-in for a model file


def spec_for(payload: bytes = PAYLOAD) -> ModelSpec:
    return replace(
        DEFAULT_MODEL,
        filename="tiny.gguf",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status


class FakeServer:
    """Records requests; serves PAYLOAD honouring (or ignoring) Range headers."""

    def __init__(self, payload: bytes = PAYLOAD, *, honour_range: bool = True) -> None:
        self.payload = payload
        self.honour_range = honour_range
        self.requests: list[urllib.request.Request] = []
        self.fail_with: Exception | None = None

    def __call__(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        if self.fail_with:
            raise self.fail_with
        range_header = request.get_header("Range")
        if range_header and self.honour_range:
            start = int(range_header.removeprefix("bytes=").rstrip("-"))
            return FakeResponse(self.payload[start:], 206)
        return FakeResponse(self.payload, 200)


def write_model(path: Path, data: bytes = PAYLOAD) -> Path:
    path.write_bytes(data)
    return path


def test_check_missing_file_has_actionable_hint(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match="not found") as info:
        check_model_file(tmp_path / "nope.gguf")
    assert info.value.hint and "jarvis model download" in info.value.hint


def test_check_rejects_non_gguf(tmp_path: Path) -> None:
    with pytest.raises(ModelFileError, match="not a GGUF"):
        check_model_file(write_model(tmp_path / "x.gguf", b"PK\x03\x04zip"))


def test_check_size_and_hash(tmp_path: Path) -> None:
    spec = spec_for()
    path = write_model(tmp_path / "tiny.gguf")
    assert check_model_file(path, spec).sha256_ok is None
    assert check_model_file(path, spec, verify_hash=True).sha256_ok is True
    with pytest.raises(ModelFileError, match="should be"):
        check_model_file(write_model(tmp_path / "short.gguf", PAYLOAD[:-1]), spec)
    tampered = PAYLOAD[:-1] + b"\x00"
    with pytest.raises(ModelFileError, match="SHA-256"):
        check_model_file(write_model(tmp_path / "t.gguf", tampered), spec, verify_hash=True)


def test_download_fresh(tmp_path: Path) -> None:
    server = FakeServer()
    seen: list[int] = []
    dest = download_model(
        spec_for(),
        tmp_path / "m" / "tiny.gguf",
        opener=server,
        progress=lambda d, t: seen.append(d),
    )
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "m" / "tiny.gguf.part").exists()
    assert seen[-1] == len(PAYLOAD)
    request = server.requests[0]
    assert request.full_url.startswith("https://huggingface.co/")
    assert request.get_header("Range") is None
    assert "jarvis-assistant" in (request.get_header("User-agent") or "")


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    dest = tmp_path / "tiny.gguf"
    (tmp_path / "tiny.gguf.part").write_bytes(PAYLOAD[:1000])
    server = FakeServer()
    download_model(spec_for(), dest, opener=server)
    assert server.requests[0].get_header("Range") == "bytes=1000-"
    assert dest.read_bytes() == PAYLOAD


def test_download_restarts_if_server_ignores_range(tmp_path: Path) -> None:
    dest = tmp_path / "tiny.gguf"
    (tmp_path / "tiny.gguf.part").write_bytes(b"garbage!" * 10)
    download_model(spec_for(), dest, opener=FakeServer(honour_range=False))
    assert dest.read_bytes() == PAYLOAD


def test_download_checksum_mismatch_deletes_partial(tmp_path: Path) -> None:
    bad = PAYLOAD[:-1] + b"\x00"
    with pytest.raises(ModelFileError, match="SHA-256"):
        download_model(spec_for(), tmp_path / "tiny.gguf", opener=FakeServer(bad))
    assert not (tmp_path / "tiny.gguf.part").exists()
    assert not (tmp_path / "tiny.gguf").exists()


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (urllib.error.HTTPError("u", 404, "Not Found", None, None), "HTTP 404"),  # type: ignore[arg-type]
        (urllib.error.URLError("network down"), "interrupted"),
        (TimeoutError("timed out"), "interrupted"),
    ],
)
def test_download_errors(tmp_path: Path, error: Exception, match: str) -> None:
    server = FakeServer()
    server.fail_with = error
    with pytest.raises(ModelFileError, match=match):
        download_model(spec_for(), tmp_path / "tiny.gguf", opener=server)


def test_download_refuses_plain_http(tmp_path: Path) -> None:
    class Insecure(ModelSpec):
        @property
        def url(self) -> str:
            return "http://example.com/model.gguf"

    insecure: Any = Insecure(**{**spec_for().__dict__})
    with pytest.raises(ModelFileError, match="non-HTTPS"):
        download_model(insecure, tmp_path / "tiny.gguf", opener=FakeServer())


def test_sha256_file(tmp_path: Path) -> None:
    path = write_model(tmp_path / "x.gguf")
    assert sha256_file(path) == hashlib.sha256(PAYLOAD).hexdigest()
