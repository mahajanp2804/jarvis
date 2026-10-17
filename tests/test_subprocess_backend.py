from __future__ import annotations

import io
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from jarvis.ai.base import Completion
from jarvis.ai.llama_cpp_backend import ModelLoadError
from jarvis.ai.model_files import ModelFileError
from jarvis.ai.subprocess_backend import SubprocessBackend
from jarvis.ai.worker import serve
from tests.fakes import FakeLLM

FAKE_WORKER = str(Path(__file__).with_name("fake_worker.py"))
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "custom.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 32)
    return path


def backend(model_file: Path, mode: str = "ok", **kwargs: float) -> SubprocessBackend:
    return SubprocessBackend(
        {"model_path": str(model_file)},
        command=[sys.executable, FAKE_WORKER, mode],
        **kwargs,
    )


def test_load_complete_unload(model_file: Path) -> None:
    b = backend(model_file)
    assert not b.is_loaded and b.process_ids() == []
    b.load()
    assert b.is_loaded
    pid = b.process_ids()[0]
    completion = b.complete_json(MESSAGES, {}, max_tokens=10, temperature=0)  # type: ignore[arg-type]
    assert json.loads(completion.text) == {"echo": "hi"}
    assert completion.prompt_tokens == 10
    b.unload()
    assert not b.is_loaded and b.process_ids() == []
    with pytest.raises(ProcessLookupError):
        import os

        os.kill(pid, 0)  # the worker process is gone: all its memory is returned


def test_missing_model_fails_before_spawning(tmp_path: Path) -> None:
    b = backend(tmp_path / "missing.gguf")
    with pytest.raises(ModelFileError, match="not found"):
        b.load()
    assert b.process_ids() == []


def test_worker_errors_keep_their_type_and_hint(model_file: Path) -> None:
    b = backend(model_file, "fail_load")
    with pytest.raises(ModelFileError) as info:
        b.load()
    assert info.value.hint == "download it"
    assert not b.is_loaded


def test_worker_crash_becomes_a_clear_error(model_file: Path) -> None:
    b = backend(model_file, "crash")
    b.load()
    with pytest.raises(ModelLoadError, match="stopped"):
        b.complete_json(MESSAGES, {}, max_tokens=10, temperature=0)  # type: ignore[arg-type]
    assert not b.is_loaded


def test_hung_worker_times_out_and_is_killed(model_file: Path) -> None:
    b = backend(model_file, "hang", request_timeout=0.5)
    b.load()
    started = time.monotonic()
    with pytest.raises(ModelLoadError, match="did not answer"):
        b.complete_json(MESSAGES, {}, max_tokens=10, temperature=0)  # type: ignore[arg-type]
    assert time.monotonic() - started < 3
    assert not b.is_loaded


def test_abort_interrupts_an_in_flight_request(model_file: Path) -> None:
    b = backend(model_file, "hang")
    b.load()
    errors: list[Exception] = []

    def call() -> None:
        try:
            b.complete_json(MESSAGES, {}, max_tokens=10, temperature=0)  # type: ignore[arg-type]
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    time.sleep(0.2)
    started = time.monotonic()
    b.abort()
    thread.join(timeout=5)
    assert time.monotonic() - started < 2
    assert isinstance(errors[0], ModelLoadError)
    assert not b.is_loaded


def test_worker_protocol_serve_with_fake_backend() -> None:
    llm = FakeLLM('{"action": "respond"}')
    requests = io.StringIO(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "op": "complete",
                    "messages": MESSAGES,
                    "schema": {},
                    "max_tokens": 5,
                    "temperature": 0,
                },
                {"op": "load"},
                {
                    "op": "complete",
                    "messages": MESSAGES,
                    "schema": {},
                    "max_tokens": 5,
                    "temperature": 0,
                },
                {"op": "nope"},
                {"op": "close"},
            ]
        )
        + "\n"
    )
    replies = io.StringIO()
    serve(llm, requests, replies)  # type: ignore[arg-type]
    lines = [json.loads(x) for x in replies.getvalue().splitlines()]
    assert lines[0]["ok"] is False  # complete before load -> contained error
    assert lines[1] == {"ok": True}
    assert Completion(**lines[2]["completion"]).text == '{"action": "respond"}'
    assert lines[3]["error_type"] == "ProtocolError"
    assert not llm.loaded  # unloaded on close
