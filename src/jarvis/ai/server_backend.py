"""Backend for a local model server with an OpenAI-compatible API.

``[ai] backend = "server"`` sends planning requests to a model server you run
yourself: llama.cpp's ``llama-server``, Ollama, LM Studio and similar. That
lets several programs (Jarvis, and later Hermes Agent) share one loaded model
instead of each loading its own.

* Plans stay schema-constrained: the plan schema goes in the standard
  ``response_format`` (``json_schema``), which ``llama-server`` turns into the
  same kind of grammar the built-in backend uses. Jarvis still validates every
  plan with pydantic.
* Jarvis doesn't start, stop or unload the server, so its memory use is the
  server's business. ``load()`` only checks that it answers.
* Prompts never leave the Mac: the URL must point at this machine unless
  ``[ai] allow_remote_server`` says otherwise (checked in the config).
* Standard library only (``urllib``), no extra dependency.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from jarvis.ai.base import ChatMessage, Completion, LLMBackend
from jarvis.ai.llama_cpp_backend import ModelLoadError
from jarvis.errors import JarvisError

Opener = Callable[..., Any]

START_HINT = (
    "Start the server first, e.g. `llama-server -m MODEL.gguf --jinja -c 4096 --port 8080`, "
    'or set [ai] backend = "llama_cpp" to use the built-in model.'
)


class ServerError(JarvisError):
    """The model server answered with an error or something unusable."""


class ServerBackend(LLMBackend):
    name = "server"

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        timeout: float = 180.0,
        opener: Opener = urllib.request.urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout
        self._open = opener
        self._connected = False
        self._aborted = threading.Event()

    @property
    def is_loaded(self) -> bool:
        return self._connected

    def describe(self) -> str:
        return f"model server at {self.base_url}" + (f" ({self.model})" if self.model else "")

    def load(self) -> None:
        """Check the server answers, and learn its model name if none is configured."""
        self._aborted.clear()
        try:
            data = self._request("GET", "/models", None, timeout=5.0)
        except ModelLoadError as exc:
            raise ModelLoadError(exc.message, hint=START_HINT) from exc
        except ServerError as exc:
            raise ModelLoadError(
                f"The model server at {self.base_url} isn't usable: {exc.message}",
                hint=START_HINT,
            ) from exc
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if not self.model and models and isinstance(models[0], str):
            self.model = models[0]
        self._connected = True

    def unload(self) -> None:
        # The server keeps its own model loaded; Jarvis only stops using it.
        self._connected = False

    def abort(self) -> None:
        """A request can't be cut short over HTTP; its answer is discarded instead."""
        self._aborted.set()

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if not self._connected:
            raise ModelLoadError("The model server isn't connected.")
        body = {
            "model": self.model or "local",
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": True, "schema": schema},
            },
        }
        started = time.monotonic()
        data = self._request("POST", "/chat/completions", body, timeout=self._timeout)
        seconds = time.monotonic() - started
        if self._aborted.is_set():
            raise ModelLoadError("The model request was stopped.")
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ServerError("The model server's reply has no message.") from exc
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}  # llama-server's own measurements
        details = usage.get("prompt_tokens_details") or {}
        cached = timings.get("cache_n", details.get("cached_tokens"))
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            seconds=seconds,
            prompt_eval_seconds=_ms(timings.get("prompt_ms")),
            generation_seconds=_ms(timings.get("predicted_ms")),
            cached_prompt_tokens=int(cached) if cached is not None else None,
        )

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None, *, timeout: float
    ) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(  # noqa: S310 - http(s) only, checked in the config
            self.base_url + path,
            data=payload,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._open(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300].strip()
            raise ServerError(f"The model server answered {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ModelLoadError(
                f"No answer from the model server at {self.base_url} ({reason})."
            ) from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ServerError("The model server's reply isn't JSON.") from exc
        if not isinstance(data, dict):
            raise ServerError("The model server's reply isn't a JSON object.")
        return data


def _ms(value: Any) -> float | None:
    return float(value) / 1000 if isinstance(value, int | float) else None
