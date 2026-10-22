"""Saved workflows: replay the steps of an earlier task.

"save that as morning setup" saves the completed steps of the previous
request (exact arguments, in order). "run the morning setup workflow" replays
them through the dispatcher, so validation, the permission policy,
confirmations, verification and the audit log apply to every step, every
time. A saved workflow carries no approvals.

A replay doesn't use the model and doesn't adapt: it stops at the first step
that fails. Once a replayed step returns outside content, later steps run
tainted (every change asks first), exactly as in a live request.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jarvis.agent.router import normalise_command
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.memory.store import WorkflowStep

# "work ?flow": speech-to-text sometimes hears "work flow" (measured with an Australian voice).
_SAVE = re.compile(
    r"^save\s+(?:that|this|it|the last (?:task|request))\s+(?:as\s+)?(?:an?\s+)?"
    r"(?:work ?flow\s+)?(?:called\s+|named\s+)?(?P<name>.+)$",
    re.IGNORECASE,
)
_RUN = re.compile(
    r"^(?:run|start|do|replay)\s+(?:the\s+|my\s+)?"
    r"(?:work ?flow\s+(?:called\s+|named\s+)?(?P<a>.+)|(?P<b>.+?)\s+work ?flow)$",
    re.IGNORECASE,
)
_LIST = re.compile(
    r"^(?:(?:list|show)(?:\s+(?:my|all|the|saved))*\s+work ?flows|"
    r"what work ?flows (?:do i have|are there))$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WorkflowCommand:
    action: Literal["save", "run", "list"]
    name: str = ""


def parse_workflow_command(text: str) -> WorkflowCommand | None:
    """Recognise "save that as X", "run the X workflow" and "list workflows"."""
    command = normalise_command(text)
    if _LIST.match(command):
        return WorkflowCommand("list")
    if match := _SAVE.match(command):
        return WorkflowCommand("save", match.group("name").strip(" \"'"))
    if match := _RUN.match(command):
        return WorkflowCommand("run", (match.group("a") or match.group("b")).strip(" \"'"))
    return None


def steps_to_save(steps: Sequence[tuple[ToolCall, ToolResult]]) -> list[WorkflowStep]:
    """The completed steps of a task, minus memory bookkeeping."""
    return [
        WorkflowStep(call.tool, dict(call.arguments))
        for call, result in steps
        if result.status is ToolStatus.SUCCESS and not call.tool.startswith("memory.")
    ]
