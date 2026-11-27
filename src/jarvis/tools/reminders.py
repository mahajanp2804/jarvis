"""Reminders tools: add a reminder, list open ones, mark one done.

* Reminders go to the Reminders app, so they sync through iCloud and alert on
  your iPhone too. (``timers.alarm`` rings on this Mac only.)
* Adding and completing are LOW risk, like Siri: easy to undo, nothing is
  lost. After the request has read outside content, they ask first.
* Days and times are parsed by Jarvis ("friday at 1", "Dec 3"), and the reply
  shows the date understood.
* Reminder titles come from the app, and lists can be shared with other
  people, so results that show them are marked as outside content.
"""

from __future__ import annotations

import difflib
import threading
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.dates import DateParseError, When, describe_when, parse_when
from jarvis.errors import NeedsClarificationError, ToolExecutionError, ToolInputError
from jarvis.platforms.base import PlatformAdapter, Reminder, RemindersApp
from jarvis.tools.base import PreparedAction, ToolSpec

MAX_LISTED = 30


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200, description="what to remember")
    when: str | None = Field(
        default=None, max_length=80, description='e.g. "tomorrow at 6pm", "friday"'
    )
    list: str | None = Field(default=None, max_length=100, description="a reminder list")


class ListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    list: str | None = Field(default=None, max_length=100, description="a reminder list")


class CompleteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200, description="the reminder's title")


def _when_text(reminder: Reminder, now: datetime) -> str:
    if reminder.due is None:
        return ""
    local = reminder.due.astimezone()
    when = When(local.date(), None if reminder.all_day else local.time())
    return describe_when(when, now.astimezone().date())


class ReminderTools:
    """Builds the ``reminders.*`` tool specs."""

    def __init__(
        self,
        platform: PlatformAdapter,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._platform = platform
        self._clock = clock

    def _app(self) -> RemindersApp:
        return self._platform.reminders()  # UnsupportedPlatformError off macOS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("reminders.add", "Add a reminder to the Reminders app (syncs to iPhone).",
                     AddArgs, RiskLevel.LOW, self._prepare_add),
            ToolSpec("reminders.list", "List open reminders.", ListArgs, RiskLevel.READ_ONLY,
                     self._prepare_list, idempotent=True),
            ToolSpec("reminders.complete", "Mark a reminder as done.", CompleteArgs,
                     RiskLevel.LOW, self._prepare_complete),
        ]  # fmt: skip

    @staticmethod
    def _resolve_list(app: RemindersApp, wanted: str) -> str:
        names = app.lists()
        by_key = {n.casefold(): n for n in names}
        key = wanted.casefold().removesuffix(" list").strip()
        if key in by_key:
            return by_key[key]
        close = difflib.get_close_matches(key, list(by_key), n=2, cutoff=0.75)
        if close:
            options = " or ".join(f"“{by_key[c]}”" for c in close)
            raise NeedsClarificationError(
                f"There's no list called “{wanted}”. Did you mean {options}?"
            )
        raise ToolExecutionError(
            f"There's no reminder list called “{wanted}”.",
            hint="Your lists: " + ", ".join(names) if names else None,
        )

    def _prepare_add(self, args: AddArgs) -> PreparedAction:
        app = self._app()
        title = " ".join(args.title.split())
        now = self._clock()
        when: When | None = None
        if args.when and args.when.strip():
            try:
                when = parse_when(args.when, now)
            except DateParseError as exc:
                raise ToolInputError(str(exc), hint='e.g. "tomorrow at 6pm"') from exc
            if when is None:
                raise ToolInputError(
                    f"“{args.when}” isn't a day or time.", hint='e.g. "friday at 1", "Dec 3"'
                )
        when_text = f" for {describe_when(when, now.date())}" if when else ""
        wanted_list = args.list

        def run(cancel: threading.Event) -> ToolResult:
            list_name = self._resolve_list(app, wanted_list) if wanted_list else None
            due = when.start() if when else None
            added = app.add(title, list_name, due, all_day=bool(when and when.at is None))
            where = f" to {added.list_name}" if added.list_name else ""
            return ToolResult(
                tool="reminders.add",
                status=ToolStatus.SUCCESS,
                message=f"Added the reminder “{title}”{where}{when_text}.",
                data={"id": added.id, "due": due.isoformat() if due else None},
            )

        return PreparedAction(description=f"Add the reminder “{title}”{when_text}", run=run)

    def _prepare_list(self, args: ListArgs) -> PreparedAction:
        app = self._app()
        wanted_list = args.list

        def run(cancel: threading.Event) -> ToolResult:
            list_name = self._resolve_list(app, wanted_list) if wanted_list else None
            found = app.open_reminders(list_name)
            now = self._clock()
            found.sort(key=lambda r: (r.due is None, r.due or now, r.title.casefold()))
            lines = []
            for index, reminder in enumerate(found[:MAX_LISTED], start=1):
                details = [d for d in (_when_text(reminder, now), reminder.list_name) if d]
                lines.append(
                    f"  {index}. {reminder.title}" + (f" ({', '.join(details)})" if details else "")
                )
            more = f"\n  … and {len(found) - MAX_LISTED} more" if len(found) > MAX_LISTED else ""
            where = f" in {list_name}" if list_name else ""
            message = (
                f"Open reminders{where} ({len(found)}):\n" + "\n".join(lines) + more
                if found
                else f"No open reminders{where}."
            )
            return ToolResult(
                tool="reminders.list",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"count": len(found)},
                untrusted=bool(found),
            )

        return PreparedAction(description="Read your open reminders", run=run)

    def _prepare_complete(self, args: CompleteArgs) -> PreparedAction:
        app = self._app()
        wanted = " ".join(args.title.split())

        def run(cancel: threading.Event) -> ToolResult:
            open_ones = app.open_reminders()
            key = wanted.casefold()
            exact = [r for r in open_ones if r.title.casefold() == key]
            matches = exact or [r for r in open_ones if key in r.title.casefold()]
            if not matches:
                raise ToolExecutionError(f"There's no open reminder matching “{wanted}”.")
            if len(matches) > 1:
                options = ", ".join(f"“{r.title}” ({r.list_name})" for r in matches[:5])
                raise NeedsClarificationError(f"Which one: {options}?")
            app.complete(matches[0].id)
            return ToolResult(
                tool="reminders.complete",
                status=ToolStatus.SUCCESS,
                message=f"Marked “{matches[0].title}” as done.",
                data={"id": matches[0].id},
                untrusted=True,  # the title comes from the app
            )

        return PreparedAction(description=f"Mark the reminder “{wanted}” as done", run=run)
