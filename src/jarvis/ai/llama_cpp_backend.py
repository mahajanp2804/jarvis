"""Local inference with llama-cpp-python (llama.cpp, Metal on Apple Silicon).

* ``llama_cpp`` is imported only inside :meth:`LlamaCppBackend.load`, so Jarvis
  starts quickly and works without the optional ``llm`` extra installed.
* The model file is memory-mapped. On Apple Silicon the weights stay file-backed
  and are shared with the GPU, so unloading frees the KV cache and compute
  buffers immediately and lets macOS reclaim the mapped pages.
* Output is constrained with a GBNF grammar generated from the JSON schema, so
  the model cannot produce text outside the schema's shape.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jarvis.ai.base import ChatMessage, Completion, LLMBackend
from jarvis.ai.model_files import check_model_file
from jarvis.ai.registry import spec_for_filename
from jarvis.errors import JarvisError

log = logging.getLogger(__name__)


class ModelLoadError(JarvisError):
    """llama.cpp could not load the model."""


class AIUnavailableError(JarvisError):
    """The local inference library is not installed."""


class LlamaCppBackend(LLMBackend):
    """:class:`LLMBackend` backed by ``llama_cpp.Llama``."""

    name = "llama_cpp"

    def __init__(
        self,
        model_path: Path,
        *,
        context_length: int = 4096,
        gpu_layers: int = -1,
        threads: int | None = None,
        chat_format: str | None = None,
        seed: int = 0,
        metal_residency_sets: bool = False,
    ) -> None:
        self.model_path = model_path
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.chat_format = chat_format
        self.seed = seed
        self.metal_residency_sets = metal_residency_sets
        self._llm: Any = None
        self._llama_cpp: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def describe(self) -> str:
        return f"llama.cpp: {self.model_path.name} (ctx {self.context_length})"

    def load(self) -> None:
        if self._llm is not None:
            return
        check_model_file(self.model_path, spec_for_filename(self.model_path.name))
        if not self.metal_residency_sets:
            # Must be set before ggml's Metal backend initialises. setdefault keeps
            # an explicit user setting. See AIConfig.metal_residency_sets.
            os.environ.setdefault("GGML_METAL_NO_RESIDENCY", "1")
        try:
            import llama_cpp
        except ImportError as exc:
            raise AIUnavailableError(
                "llama-cpp-python is not installed, so the local AI model can't run.",
                hint='In the Jarvis virtualenv run: CMAKE_ARGS="-DGGML_METAL=on" '
                'pip install -e ".[llm]"',
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.context_length,
            "n_gpu_layers": self.gpu_layers,
            "seed": self.seed,
            "use_mmap": True,
            "verbose": False,
        }
        if self.threads is not None:
            kwargs["n_threads"] = self.threads
        if self.chat_format is not None:
            kwargs["chat_format"] = self.chat_format

        started = time.monotonic()
        try:
            self._llm = llama_cpp.Llama(**kwargs)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ModelLoadError(
                f"llama.cpp could not load {self.model_path.name}: {exc}",
                hint="Check the file with `jarvis model check --verify`, or lower "
                "[ai] context_length if memory is tight.",
            ) from exc
        self._llama_cpp = llama_cpp
        log.info("Loaded %s in %.2fs", self.describe(), time.monotonic() - started)

    def unload(self) -> None:
        if self._llm is None:
            return
        llm, self._llm = self._llm, None
        try:
            llm.close()
        finally:
            del llm
            gc.collect()
        log.info("Unloaded %s", self.model_path.name)

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if self._llm is None:
            raise ModelLoadError("The model is not loaded.")
        grammar = self._llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        ctx = self._perf_context()
        if ctx is not None:
            self._llama_cpp.llama_perf_context_reset(ctx)

        started = time.monotonic()
        response = self._llm.create_chat_completion(
            messages=list(messages),
            grammar=grammar,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=self.seed,
        )
        seconds = time.monotonic() - started

        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        prompt_eval_s = generation_s = None
        cached: int | None = None
        if ctx is not None:
            perf = self._llama_cpp.llama_perf_context(ctx)
            prompt_eval_s = perf.t_p_eval_ms / 1000
            generation_s = perf.t_eval_ms / 1000
            # Tokens not re-evaluated because they matched the previous prompt.
            cached = max(prompt_tokens - int(perf.n_p_eval), 0)
        return Completion(
            text=response["choices"][0]["message"]["content"] or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage.get("completion_tokens", 0)),
            seconds=seconds,
            prompt_eval_seconds=prompt_eval_s,
            generation_seconds=generation_s,
            cached_prompt_tokens=cached,
        )

    def _perf_context(self) -> Any:
        # Internal llama-cpp-python attribute; perf numbers are optional extras.
        try:
            return self._llm._ctx.ctx
        except AttributeError:
            return None
