"""LLM backend that runs the model in a child process (see :mod:`jarvis.ai.worker`).

``load()`` starts the worker and waits until the model is loaded; ``unload()``
ends it, which returns all of the model's memory to the OS. ``abort()`` kills
the worker from any thread, so an in-flight request fails fast with a
:class:`ModelLoadError` instead of blocking shutdown.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from jarvis.ai.base import ChatMessage, Completion, LLMBackend
from jarvis.ai.llama_cpp_backend import AIUnavailableError, ModelLoadError
from jarvis.ai.model_files import ModelFileError, check_model_file
from jarvis.ai.registry import spec_for_filename
from jarvis.errors import JarvisError

log = logging.getLogger(__name__)

_ERRORS: dict[str, type[JarvisError]] = {
    "ModelFileError": ModelFileError,
    "AIUnavailableError": AIUnavailableError,
    "ModelLoadError": ModelLoadError,
}
_EOF = None


class SubprocessBackend(LLMBackend):
    """:class:`LLMBackend` that proxies to a model worker process."""

    name = "llama_cpp (worker process)"

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        command: Sequence[str] | None = None,
        load_timeout: float = 120.0,
        request_timeout: float = 180.0,
    ) -> None:
        self.settings = settings
        self.model_path = Path(settings["model_path"])
        self._command = (
            list(command)
            if command
            else [
                sys.executable,
                "-m",
                "jarvis.ai.worker",
                json.dumps(settings),
            ]
        )
        self._load_timeout = load_timeout
        self._request_timeout = request_timeout
        self._proc: subprocess.Popen[str] | None = None
        self._replies: queue.Queue[str | None] = queue.Queue()
        self._proc_lock = threading.Lock()  # guards _proc against abort() races

    @property
    def is_loaded(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def describe(self) -> str:
        ctx = self.settings.get("context_length", "?")
        return f"llama.cpp worker: {self.model_path.name} (ctx {ctx})"

    def process_ids(self) -> list[int]:
        proc = self._proc
        return [proc.pid] if proc is not None and proc.poll() is None else []

    def load(self) -> None:
        if self.is_loaded:
            return
        # Fail fast (and without spawning anything) on a missing or broken file.
        check_model_file(self.model_path, spec_for_filename(self.model_path.name))
        self._replies = queue.Queue()
        try:
            proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise ModelLoadError(f"Could not start the model worker: {exc.strerror}") from exc
        with self._proc_lock:
            self._proc = proc
        if proc.stdout is None:  # pragma: no cover - stdout=PIPE guarantees a stream
            raise ModelLoadError("Model worker has no output stream.")
        threading.Thread(
            target=_pump,
            args=(proc.stdout, self._replies),
            name="jarvis-worker-reader",
            daemon=True,
        ).start()
        try:
            self._request({"op": "load"}, self._load_timeout)
        except BaseException:
            self._kill()
            raise
        log.info("Model worker %d loaded %s", proc.pid, self.model_path.name)

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if not self.is_loaded:
            raise ModelLoadError("The model is not loaded.")
        reply = self._request(
            {
                "op": "complete",
                "messages": list(messages),
                "schema": schema,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            self._request_timeout,
        )
        return Completion(**reply["completion"])

    def unload(self) -> None:
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait(timeout=5)
        log.info("Model worker %d stopped", proc.pid)

    def abort(self) -> None:
        """Kill the worker immediately (safe from any thread)."""
        self._kill()

    # -- internals -------------------------------------------------------------

    def _kill(self) -> None:
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            log.info("Model worker %d killed", proc.pid)

    def _request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ModelLoadError("The model worker is not running.")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ModelLoadError("The model worker stopped unexpectedly.") from exc
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise ModelLoadError(f"The model worker did not answer within {timeout:g}s.") from None
        if line is _EOF:
            code = proc.poll()
            self._kill()
            raise ModelLoadError(
                "The model worker stopped" + (f" (exit code {code})." if code is not None else ".")
            )
        reply: dict[str, Any] = json.loads(line)
        if not reply.get("ok"):
            error_cls = _ERRORS.get(str(reply.get("error_type")), ModelLoadError)
            raise error_cls(
                str(reply.get("message", "Model worker error.")), hint=reply.get("hint")
            )
        return reply


def _pump(stream: IO[str], replies: queue.Queue[str | None]) -> None:
    """Move worker replies into a queue so reads can time out (portable, no select)."""
    try:
        for line in stream:
            replies.put(line)
    except (OSError, ValueError):
        pass
    finally:
        replies.put(_EOF)
