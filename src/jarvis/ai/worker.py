"""Model worker process: hosts one :class:`LlamaCppBackend` for the main process.

Running inference in a child process means:

* **True unload.** Ending the worker returns *all* model memory to the OS,
  including the ~130 MB of Metal/llama.cpp runtime state that stays in a
  process after an in-process unload.
* **Abortable.** A running generation can be stopped by killing the worker, so
  shutting Jarvis down never waits for a long native call.
* **Crash isolation.** A native crash in llama.cpp ends the worker, not Jarvis.

Protocol: one JSON object per line on stdin/stdout.

    -> {"op": "load"}                                   <- {"ok": true}
    -> {"op": "complete", "messages": [...], "schema": {...},
        "max_tokens": n, "temperature": t}              <- {"ok": true, "completion": {...}}
    -> {"op": "close"}                                  (worker exits)
    errors                                              <- {"ok": false, "error_type": ...,
                                                            "message": ..., "hint": ...}

Backend settings arrive as one JSON argument (paths and numbers only).
The worker exits when stdin closes, so it can never outlive its parent.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from jarvis.ai.llama_cpp_backend import LlamaCppBackend
from jarvis.ai.prompt_cache import PromptCache
from jarvis.errors import JarvisError

log = logging.getLogger("jarvis.ai.worker")


def backend_from_settings(settings: dict[str, Any]) -> LlamaCppBackend:
    cache_dir = settings.get("prompt_cache_dir")
    prompt_cache = (
        PromptCache(Path(cache_dir), int(settings.get("prompt_cache_max_mb", 512)) << 20)
        if cache_dir
        else None
    )
    return LlamaCppBackend(
        Path(settings["model_path"]),
        context_length=int(settings.get("context_length", 4096)),
        gpu_layers=int(settings.get("gpu_layers", -1)),
        threads=settings.get("threads"),
        chat_format=settings.get("chat_format"),
        metal_residency_sets=bool(settings.get("metal_residency_sets", False)),
        prompt_cache=prompt_cache,
    )


def _protocol_stream() -> TextIO:
    """Reserve the real stdout for the protocol; send fd 1 to stderr.

    Native libraries may print to file descriptor 1, which would corrupt the
    JSON stream, so everything except protocol replies goes to stderr.
    """
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def serve(backend: LlamaCppBackend, requests: TextIO, replies: TextIO) -> None:
    """Answer requests until ``close`` or end of input."""
    for line in requests:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "close":
                break
            if op == "load":
                backend.load()
                reply: dict[str, Any] = {"ok": True}
            elif op == "complete":
                completion = backend.complete_json(
                    request["messages"],
                    request["schema"],
                    max_tokens=int(request["max_tokens"]),
                    temperature=float(request["temperature"]),
                )
                reply = {"ok": True, "completion": asdict(completion)}
            else:
                reply = {"ok": False, "error_type": "ProtocolError", "message": f"bad op {op!r}"}
        except JarvisError as exc:
            reply = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": exc.message,
                "hint": exc.hint,
            }
        except Exception as exc:
            log.exception("Model worker request failed")
            reply = {"ok": False, "error_type": "InternalError", "message": str(exc)[:500]}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()
    backend.unload()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m jarvis.ai.worker '<settings json>'", file=sys.stderr)
        return 2
    # The parent coordinates shutdown; a terminal Ctrl+C must not kill us first.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=logging.WARNING, stream=sys.stderr, format="jarvis-worker: %(message)s"
    )
    proto = _protocol_stream()
    serve(backend_from_settings(json.loads(args[0])), sys.stdin, proto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
