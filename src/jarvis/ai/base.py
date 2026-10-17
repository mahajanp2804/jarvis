"""Model backend interface.

A backend turns chat messages into JSON that is *constrained* to a JSON schema
(grammar-based decoding for local models, structured-output APIs for optional
cloud backends later). Callers still validate the JSON with pydantic, because
constrained decoding guarantees shape, not meaning.

Backends own the model's memory. ``load``/``unload`` let the
:class:`~jarvis.ai.model_manager.ModelManager` keep a local model resident only
while it is useful. Remote backends can implement them as no-ops.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class ChatMessage(TypedDict):
    """One chat turn in OpenAI-style format."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Completion:
    """Generated text plus the measurements the benchmark reports."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    # Breakdown when the backend can report it (llama.cpp perf counters).
    prompt_eval_seconds: float | None = None
    generation_seconds: float | None = None
    cached_prompt_tokens: int | None = None


class LLMBackend(ABC):
    """A language model that produces schema-constrained JSON."""

    name: str

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model currently occupies memory."""

    @abstractmethod
    def load(self) -> None:
        """Load the model. Raise a JarvisError with a hint if that is impossible."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model's memory. Must be safe to call when not loaded."""

    @abstractmethod
    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        """Generate one reply constrained to ``schema``. Requires ``load()`` first."""

    def describe(self) -> str:
        """Short human-readable description for logs and ``jarvis model info``."""
        return self.name

    def abort(self) -> None:
        """Interrupt an in-flight request from another thread, if the backend can.

        Called during shutdown. The default does nothing (an in-process llama.cpp
        call cannot be interrupted); the worker-process backend kills its worker.
        """
        return None

    def process_ids(self) -> list[int]:
        """Extra OS processes holding the model (for resource measurements)."""
        return []
