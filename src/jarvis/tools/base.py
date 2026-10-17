"""Tool abstractions.

A tool runs in two stages so that confirmations and dry runs describe exactly
what will happen:

1. ``prepare(args)`` validates and resolves the request (e.g. "safari" ->
   ``/Applications/Safari.app``) and applies resource-level policy. It must
   not change anything on the system.
2. ``PreparedAction.run(cancel)`` performs the side effect and verifies it.

The dispatcher calls ``prepare``, consults the policy, optionally asks for
confirmation using ``PreparedAction.description``, and only then calls ``run``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from jarvis.api.schemas import RiskLevel, ToolResult
from jarvis.errors import UnknownToolError


@dataclass(frozen=True)
class PreparedAction:
    """A validated, fully resolved action that is ready to run."""

    description: str  # shown in confirmations and dry runs, e.g. "Quit Safari"
    run: Callable[[threading.Event], ToolResult]
    # Optional extra context for the confirmation prompt (e.g. consequences).
    warning: str | None = None
    # True when prepare() found nothing to change; run() must then have no side
    # effects, and the dispatcher skips the confirmation prompt.
    noop: bool = False
    # Risk of this specific call, when it differs from the tool's default
    # (e.g. `git status` vs `git push` through the same terminal tool).
    risk: RiskLevel | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Static description of a tool, used by the dispatcher and (later) the planner."""

    name: str
    description: str
    args_model: type[BaseModel]
    risk: RiskLevel
    prepare: Callable[[Any], PreparedAction]
    # Safe to run twice with the same effect, so transient failures may be retried.
    idempotent: bool = False

    def json_schema(self) -> dict[str, Any]:
        """JSON schema of the arguments (used for constrained LLM output in Phase 2)."""
        return self.args_model.model_json_schema()


class ToolRegistry:
    """Name -> :class:`ToolSpec` lookup."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownToolError(f"Unknown tool '{name}'.") from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def __iter__(self) -> Any:
        return iter(self._specs.values())
