"""ServerBackend against a real local HTTP server that imitates llama-server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from jarvis.agent.planner import Planner, PlannerSettings
from jarvis.ai.llama_cpp_backend import ModelLoadError
from jarvis.ai.model_manager import ModelManager
from jarvis.ai.server_backend import ServerBackend, ServerError
from jarvis.bootstrap import Runtime
from tests.fakes import plan_json


class FakeModelServer:
    """Records requests; answers like llama-server's OpenAI-compatible API."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.reply = plan_json("respond", "Hello.")
        self.status = 200
        self.raw: bytes | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                outer.requests.append(("GET", self.path, None))
                self._send(200, json.dumps({"data": [{"id": "qwen3-4b"}]}).encode())

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                outer.requests.append(("POST", self.path, body))
                if outer.raw is not None:
                    self._send(outer.status, outer.raw)
                    return
                answer = {
                    "choices": [{"message": {"role": "assistant", "content": outer.reply}}],
                    "usage": {"prompt_tokens": 1900, "completion_tokens": 30},
                    "timings": {"prompt_ms": 120.0, "predicted_ms": 1050.0, "cache_n": 1880},
                }
                self._send(outer.status, json.dumps(answer).encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[FakeModelServer]:
    fake = FakeModelServer()
    yield fake
    fake.close()


def test_load_learns_the_model_and_requests_are_schema_constrained(
    server: FakeModelServer,
) -> None:
    backend = ServerBackend(server.url)
    assert not backend.is_loaded
    backend.load()
    assert backend.is_loaded and backend.model == "qwen3-4b"
    schema = {"type": "object", "properties": {"action": {"type": "string"}}}
    messages: list[Any] = [{"role": "user", "content": "hi"}]
    completion = backend.complete_json(messages, schema, max_tokens=64, temperature=0.0)
    method, path, body = server.requests[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body["model"] == "qwen3-4b" and body["messages"] == messages
    assert body["max_tokens"] == 64 and body["temperature"] == 0.0 and body["stream"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert completion.text == server.reply
    assert (completion.prompt_tokens, completion.completion_tokens) == (1900, 30)
    assert completion.prompt_eval_seconds == pytest.approx(0.12)
    assert completion.generation_seconds == pytest.approx(1.05)
    assert completion.cached_prompt_tokens == 1880
    backend.unload()
    assert not backend.is_loaded
    assert "127.0.0.1" in backend.describe()


def test_missing_server_says_how_to_start_one() -> None:
    backend = ServerBackend("http://127.0.0.1:9/v1")  # nothing listens on port 9
    with pytest.raises(ModelLoadError, match="No answer from the model server") as info:
        backend.load()
    assert info.value.hint and "llama-server" in info.value.hint and "--jinja" in info.value.hint


def test_server_errors_are_reported(server: FakeModelServer) -> None:
    backend = ServerBackend(server.url, model="named")
    backend.load()
    assert backend.model == "named"  # a configured name is kept
    server.status, server.raw = 500, b'{"error": "context size exceeded"}'
    with pytest.raises(ServerError, match=r"answered 500: .*context size exceeded"):
        backend.complete_json([], {}, max_tokens=8, temperature=0.0)
    server.status, server.raw = 200, b"not json"
    with pytest.raises(ServerError, match="isn't JSON"):
        backend.complete_json([], {}, max_tokens=8, temperature=0.0)
    server.raw = b'{"choices": []}'
    with pytest.raises(ServerError, match="no message"):
        backend.complete_json([], {}, max_tokens=8, temperature=0.0)


def test_abort_discards_the_answer(server: FakeModelServer) -> None:
    backend = ServerBackend(server.url)
    with pytest.raises(ModelLoadError, match="isn't connected"):
        backend.complete_json([], {}, max_tokens=8, temperature=0.0)
    backend.load()
    backend.abort()
    with pytest.raises(ModelLoadError, match="stopped"):
        backend.complete_json([], {}, max_tokens=8, temperature=0.0)
    backend.load()  # a new session starts clean
    backend.complete_json([], {}, max_tokens=8, temperature=0.0)


def test_planner_works_through_the_server(server: FakeModelServer, make_runtime: Any) -> None:
    runtime: Runtime = make_runtime()[0]
    server.reply = plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"}))
    planner = Planner(
        ModelManager(ServerBackend(server.url), idle_unload_seconds=0),
        lambda: list(runtime.registry),
        PlannerSettings(),
    )
    plan = planner.plan("open safari")
    assert [(s.tool, s.arguments) for s in plan.steps] == [("apps.open", {"name": "Safari"})]
    sent = server.requests[-1][2]
    assert (
        sent["messages"][0]["role"] == "system" and "apps.open(" in sent["messages"][0]["content"]
    )
