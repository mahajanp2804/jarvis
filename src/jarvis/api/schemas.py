"""Shared, validated data contracts between the agent, dispatcher and tools.

Every action Jarvis takes is expressed as a :class:`ToolCall` and answered with
a :class:`ToolResult`. Tool calls may originate from the rule-based router or
(from Phase 2) from a language model, so they are always treated as untrusted
input and validated before anything runs.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(IntEnum):
    """How much harm an action could cause if it were wrong.

    Ordered so policies can compare levels (``risk >= RiskLevel.MEDIUM``).
    """

    READ_ONLY = 0  # inspects state only (list apps, read a file)
    LOW = 1  # easily reversible, no data loss (open an app)
    MEDIUM = 2  # may interrupt work (quit an app)
    HIGH = 3  # may lose data or act externally (delete, send, submit)
    CRITICAL = 4  # system-wide or privileged changes

    @classmethod
    def parse(cls, value: object) -> RiskLevel:
        """Accept an enum member, its integer value, or a case-insensitive name."""
        if isinstance(value, RiskLevel):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                names = ", ".join(m.name.lower() for m in cls)
                raise ValueError(
                    f"unknown risk level '{value}' (expected one of: {names})"
                ) from None
        if isinstance(value, int) and not isinstance(value, bool):
            return cls(value)
        raise ValueError(f"invalid risk level: {value!r}")


class ToolStatus(StrEnum):
    """Outcome of a single tool call."""

    SUCCESS = "success"
    FAILED = "failed"
    DENIED = "denied"  # blocked by policy
    CANCELLED = "cancelled"  # user declined confirmation, or Jarvis is stopping
    DRY_RUN = "dry_run"  # validated and described, not executed
    NEEDS_CLARIFICATION = "needs_clarification"  # ambiguous input; ask the user


class ToolCall(BaseModel):
    """A request to run one tool with arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """What happened when a tool call was processed."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    status: ToolStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    verified: bool | None = None  # None: verification not applicable / not possible
    hint: str | None = None
    error_type: str | None = None
    # The message/data contain content from outside Jarvis (file contents, web
    # pages, command output, OCR). It is shown to the model only as data, and
    # later side-effecting steps in the same request need confirmation.
    untrusted: bool = False
    # A failure that an idempotent tool may retry (e.g. a page load timeout).
    retryable: bool = False
    attempts: int = 1

    @property
    def ok(self) -> bool:
        """True when the call completed (or was successfully dry-run)."""
        return self.status in (ToolStatus.SUCCESS, ToolStatus.DRY_RUN)
