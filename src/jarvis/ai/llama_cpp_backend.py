"""Local inference with llama-cpp-python (llama.cpp, Metal on Apple Silicon).

* ``llama_cpp`` is imported only inside :meth:`LlamaCppBackend.load`, so Jarvis
  starts quickly and works without the optional ``llm`` extra installed.
* The model file is memory-mapped. On Apple Silicon the weights stay file-backed
  and are shared with the GPU, so unloading frees the KV cache and compute
  buffers immediately and lets macOS reclaim the mapped pages.
* Output is constrained with a GBNF grammar generated from the JSON schema, so
  the model cannot produce text outside the schema's shape.
* With a :class:`~jarvis.ai.prompt_cache.PromptCache`, the KV state of the
  system prompt is saved to disk once and restored after later loads, so the
  first request after a load doesn't re-read the prompt.
"""

from __future__ import annotations

import ctypes
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
from jarvis.ai.prompt_cache import PromptCache, system_prefix_length
from jarvis.ai.registry import spec_for_filename
from jarvis.errors import JarvisError

log = logging.getLogger(__name__)

# Shorter system prompts are cheap to re-read; not worth a file.
_MIN_CACHED_PROMPT_TOKENS = 64


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
        prompt_cache: PromptCache | None = None,
    ) -> None:
        self.model_path = model_path
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.chat_format = chat_format
        self.seed = seed
        self.metal_residency_sets = metal_residency_sets
        self.prompt_cache = prompt_cache
        self._llm: Any = None
        self._llama_cpp: Any = None
        # Key of the system prompt whose state is in the KV cache (saved or restored).
        self._prompt_key: str | None = None

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
        self._prompt_key = None
        log.info("Loaded %s in %.2fs", self.describe(), time.monotonic() - started)

    def unload(self) -> None:
        if self._llm is None:
            return
        llm, self._llm = self._llm, None
        self._prompt_key = None
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
        key = self._prompt_cache_key(messages)
        cache_event: str | None = None
        cache_seconds: float | None = None
        if key is not None and key != self._prompt_key:
            t0 = time.monotonic()
            if self._restore_prompt(key):
                self._prompt_key = key
                cache_event, cache_seconds = "restored", time.monotonic() - t0

        ctx = self._native_context()
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

        if key is not None and key != self._prompt_key:
            # This request read the system prompt in full: save it for the next load.
            self._prompt_key = key  # one attempt per prompt, even if saving fails
            t0 = time.monotonic()
            if self._save_prompt(key, messages[0]["content"], prompt_tokens):
                cache_event, cache_seconds = "saved", time.monotonic() - t0
        return Completion(
            text=response["choices"][0]["message"]["content"] or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=int(usage.get("completion_tokens", 0)),
            seconds=seconds,
            prompt_eval_seconds=prompt_eval_s,
            generation_seconds=generation_s,
            cached_prompt_tokens=cached,
            prompt_cache=cache_event,
            prompt_cache_seconds=cache_seconds,
        )

    # -- prompt cache ------------------------------------------------------------

    def _prompt_cache_key(self, messages: Sequence[ChatMessage]) -> str | None:
        if self.prompt_cache is None or not messages or messages[0]["role"] != "system":
            return None
        try:
            return PromptCache.make_key(
                system_prompt=messages[0]["content"],
                model_path=self.model_path,
                context_length=self.context_length,
                chat_format=self.chat_format,
                library_version=str(getattr(self._llama_cpp, "__version__", "?")),
            )
        except OSError:
            return None

    def _restore_prompt(self, key: str) -> bool:
        """Load the saved state for ``key`` into the KV cache, if there is one."""
        ctx = self._native_context()
        if self.prompt_cache is None or ctx is None:
            return False
        path = self.prompt_cache.find(key)
        if path is None:
            return False
        lib, llm = self._llama_cpp, self._llm
        capacity = self.context_length
        tokens = (lib.llama_token * capacity)()
        count = ctypes.c_size_t(0)
        if not lib.llama_state_load_file(
            ctx, os.fsencode(path), tokens, capacity, ctypes.byref(count)
        ):
            # llama.cpp clears the KV cache when a load fails.
            log.warning("Discarding unreadable prompt cache %s", path.name)
            llm.reset()
            self.prompt_cache.discard(path)
            return False
        # The tokens end inside the system turn, so llama-cpp-python always
        # evaluates at least the request (and refreshes the logits) before sampling.
        n = count.value
        llm.input_ids[:n] = tokens[:n]
        llm.n_tokens = n
        log.info("Restored %d prompt tokens from %s", n, path.name)
        return True

    def _save_prompt(self, key: str, system_prompt: str, prompt_tokens: int) -> bool:
        """Save the state of the system prompt (not the request) for later loads."""
        ctx = self._native_context()
        if self.prompt_cache is None or ctx is None:
            return False
        lib, llm = self._llama_cpp, self._llm
        tokens = [int(t) for t in llm.input_ids[:prompt_tokens]]
        n = system_prefix_length(
            tokens, system_prompt, lambda toks: bytes(llm.detokenize(list(toks), special=True))
        )
        if n < _MIN_CACHED_PROMPT_TOKENS:
            return False
        # Drop the request and the reply from the KV cache so they aren't written
        # to disk. A follow-up round of this request re-reads them (~0.2 s, once).
        if not lib.llama_memory_seq_rm(lib.llama_get_memory(ctx), -1, n, -1):
            return False
        llm.n_tokens = n
        size = int(lib.llama_state_get_size(ctx))
        if not self.prompt_cache.fits(size):
            log.info("Prompt state (%d MB) exceeds [ai] prompt_cache_max_mb; not saved", size >> 20)
            return False
        array = (lib.llama_token * n)(*tokens[:n])

        def write(path: Path) -> bool:
            return bool(lib.llama_state_save_file(ctx, os.fsencode(path), array, n))

        saved = self.prompt_cache.store(key, write)
        if saved is not None:
            log.info("Saved %d prompt tokens to %s", n, saved.name)
        return saved is not None

    def _native_context(self) -> Any:
        # Internal llama-cpp-python attribute; perf numbers and the prompt cache
        # are optional extras.
        try:
            return self._llm._ctx.ctx
        except AttributeError:
            return None
