"""Apple Shortcuts tools: list your shortcuts and run one.

Shortcuts reach what macOS has no command for (brightness, Focus, Bluetooth,
HomeKit scenes, other apps' actions), so this one bridge covers a lot.

* **Running asks first** (MEDIUM): Jarvis can't see what a shortcut does, and
  it runs with your permissions. Shortcuts named in ``[shortcuts] trusted`` run
  without asking (LOW), except after the request has read outside content,
  when every change asks.
* Names must match an existing shortcut. A near miss ("water eject" typed as
  "water ejec") gets "Did you mean …?" instead of a guess.
* Whatever a shortcut returns is outside content: shown to the model as data,
  never as instructions.
"""

from __future__ import annotations

import difflib
import threading

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import ShortcutsConfig
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.platforms.base import PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec

_MAX_OUTPUT = 2000


class ListShortcutsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, max_length=100, description="words in the name")


class RunShortcutArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200, description="the shortcut's exact name")
    input: str | None = Field(
        default=None, max_length=2000, description="text to pass to the shortcut"
    )


class ShortcutTools:
    """Builds the ``shortcuts.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter, config: ShortcutsConfig) -> None:
        self._platform = platform
        self._config = config
        self._trusted = {name.casefold() for name in config.trusted}

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "shortcuts.list",
                "List the user's Apple Shortcuts, e.g. to find one for screen brightness, "
                "Focus or smart-home lights.",
                ListShortcutsArgs,
                RiskLevel.READ_ONLY,
                self._prepare_list,
                idempotent=True,
            ),
            ToolSpec(
                "shortcuts.run",
                "Run the user's Apple Shortcut with this exact name from shortcuts.list.",
                RunShortcutArgs,
                RiskLevel.MEDIUM,
                self._prepare_run,
            ),
        ]

    def _resolve(self, name: str) -> str:
        names = self._platform.list_shortcuts()
        by_key = {n.casefold(): n for n in names}
        if name.casefold() in by_key:
            return by_key[name.casefold()]
        close = difflib.get_close_matches(name.casefold(), list(by_key), n=2, cutoff=0.75)
        if close:
            options = " or ".join(f"“{by_key[c]}”" for c in close)
            raise NeedsClarificationError(
                f"There's no shortcut called “{name}”. Did you mean {options}?"
            )
        known = ", ".join(names[:15]) + (" …" if len(names) > 15 else "")
        raise ToolExecutionError(
            f"There's no shortcut called “{name}”.",
            hint=f"Your shortcuts: {known}." if names else "Make one in the Shortcuts app first.",
        )

    def _prepare_run(self, args: RunShortcutArgs) -> PreparedAction:
        name = self._resolve(args.name)
        given = args.input

        def run(cancel: threading.Event) -> ToolResult:
            output = self._platform.run_shortcut(
                name, given, timeout=self._config.timeout_seconds, cancel=cancel
            ).strip()
            message = f"Ran the shortcut “{name}”."
            if output:
                shown = output[:_MAX_OUTPUT] + (
                    "\n[… more output not shown]" if len(output) > _MAX_OUTPUT else ""
                )
                message += f" It returned:\n{shown}"
            return ToolResult(
                tool="shortcuts.run",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"name": name, "output_chars": len(output)},
                untrusted=bool(output),  # what a shortcut returns is outside content
            )

        preview = f" with “{given[:60]}{'…' if len(given) > 60 else ''}”" if given else ""
        trusted = name.casefold() in self._trusted
        return PreparedAction(
            description=f"Run the shortcut “{name}”{preview}",
            run=run,
            warning=None
            if trusted
            else "Jarvis can't see what a shortcut does: it runs with your permissions.",
            risk=RiskLevel.LOW if trusted else None,
        )

    def _prepare_list(self, args: ListShortcutsArgs) -> PreparedAction:
        def run(cancel: threading.Event) -> ToolResult:
            names = self._platform.list_shortcuts()
            if args.query:
                names = [n for n in names if args.query.casefold() in n.casefold()]
            if names:
                message = f"Shortcuts ({len(names)}):\n" + "\n".join(f"  {n}" for n in names)
            else:
                found = f" matching '{args.query}'" if args.query else ""
                message = f"No shortcuts{found}."
            return ToolResult(
                tool="shortcuts.list",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"shortcuts": names},
            )

        return PreparedAction(description="List your shortcuts", run=run)
