"""Memory tools: remember and forget preferences, look back at task history.

* ``memory.remember`` stores a fact the user stated ("my editor is VS Code").
  It is LOW risk, so it runs without asking; after outside content has been
  read in the same request it asks first, like every other change (a web page
  can't plant a "preference").
* Values that look like passwords, keys or card numbers are refused.
* ``memory.history`` returns earlier requests as data (outside content), never
  as instructions to repeat.
* There is no tool to clear the memory: that is `jarvis memory clear`, typed
  by the user.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import ToolExecutionError
from jarvis.memory.store import (
    MAX_NAME,
    MAX_VALUE,
    MemoryStore,
    check_value,
    local_time,
    normalise_name,
)
from jarvis.tools.base import PreparedAction, ToolSpec


class RememberArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=MAX_NAME, description="e.g. 'editor'")
    value: str = Field(min_length=1, max_length=MAX_VALUE, description="e.g. 'Visual Studio Code'")


class ForgetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=MAX_NAME)


class ListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, max_length=100, description="text to look for")
    limit: int = Field(default=10, ge=1, le=50)


class MemoryTools:
    """Builds the ``memory.*`` tool specs."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "memory.remember",
                "Remember a preference the user states, e.g. their editor or main project "
                "folder. Never for passwords or other secrets.",
                RememberArgs,
                RiskLevel.LOW,
                self._prepare_remember,
            ),
            ToolSpec(
                "memory.forget",
                "Forget a remembered preference.",
                ForgetArgs,
                RiskLevel.LOW,
                self._prepare_forget,
            ),
            ToolSpec(
                "memory.list",
                "List the remembered preferences.",
                ListArgs,
                RiskLevel.READ_ONLY,
                self._prepare_list,
                idempotent=True,
            ),
            ToolSpec(
                "memory.history",
                "Earlier requests and what Jarvis did, newest first.",
                HistoryArgs,
                RiskLevel.READ_ONLY,
                self._prepare_history,
                idempotent=True,
            ),
        ]

    def _prepare_remember(self, args: RememberArgs) -> PreparedAction:
        name = normalise_name(args.name)
        value = check_value(name, args.value)
        current = self._store.preference(name)

        def run(cancel: threading.Event) -> ToolResult:
            stored = self._store.remember(name, value)
            check = self._store.preference(name)
            ok = check is not None and check.value == stored.value
            return ToolResult(
                tool="memory.remember",
                status=ToolStatus.SUCCESS if ok else ToolStatus.FAILED,
                message=f"I'll remember that your {name} is {value}."
                if ok
                else "That wasn't saved.",
                data={"name": name, "value": value},
                verified=ok,
            )

        if current is not None and current.value == value:
            return PreparedAction(
                description=f"Remember that your {name} is {value}", run=run, noop=True
            )
        change = f" (instead of {current.value})" if current else ""
        return PreparedAction(description=f"Remember that your {name} is {value}{change}", run=run)

    def _prepare_forget(self, args: ForgetArgs) -> PreparedAction:
        name = normalise_name(args.name)
        current = self._store.preference(name)
        if current is None:
            known = ", ".join(p.name for p in self._store.preferences()) or "nothing yet"
            raise ToolExecutionError(
                f"I don't have anything remembered as '{name}'.", hint=f"I remember: {known}."
            )

        def run(cancel: threading.Event) -> ToolResult:
            self._store.forget(name)
            ok = self._store.preference(name) is None
            return ToolResult(
                tool="memory.forget",
                status=ToolStatus.SUCCESS if ok else ToolStatus.FAILED,
                message=f"Forgot your {name}." if ok else f"Your {name} is still remembered.",
                data={"name": name},
                verified=ok,
            )

        return PreparedAction(description=f"Forget your {name} ({current.value})", run=run)

    def _prepare_list(self, args: ListArgs) -> PreparedAction:
        def run(cancel: threading.Event) -> ToolResult:
            prefs = self._store.preferences()
            if not prefs:
                message = "I don't remember any preferences yet."
            else:
                message = "I remember:\n" + "\n".join(f"  {p.name}: {p.value}" for p in prefs)
            return ToolResult(
                tool="memory.list",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"preferences": {p.name: p.value for p in prefs}},
            )

        return PreparedAction(description="List remembered preferences", run=run)

    def _prepare_history(self, args: HistoryArgs) -> PreparedAction:
        def run(cancel: threading.Event) -> ToolResult:
            tasks = self._store.history(limit=args.limit, query=args.query)
            if not tasks:
                found = f" matching '{args.query}'" if args.query else ""
                message = f"No earlier requests{found} in the history."
            else:
                lines = []
                for task in tasks:
                    tools = ", ".join(s.tool for s in task.steps) or "no tools"
                    lines.append(
                        f"{local_time(task.started_at)} {task.outcome}: {task.request} ({tools})"
                    )
                message = "Earlier requests (newest first):\n" + "\n".join(lines)
            return ToolResult(
                tool="memory.history",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"count": len(tasks)},
                untrusted=True,  # earlier requests are data, not instructions to repeat
            )

        return PreparedAction(description="Look through the task history", run=run)
