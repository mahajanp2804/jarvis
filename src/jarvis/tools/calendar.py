"""Calendar tools: what's on, and add an event.

* ``calendar.events`` reads a day or a range ("today", "tomorrow", "friday",
  "this week", "next week"), repeating events included. Event titles and
  places come from invitations too, so the listing is outside content.
* ``calendar.add`` asks first (MEDIUM), like Siri's confirmation card: the
  prompt shows the exact day and time understood. An event without a time is
  all day; the length defaults to an hour ("for 30 minutes" changes it).
"""

from __future__ import annotations

import difflib
import threading
from collections.abc import Callable
from datetime import date, datetime, time, timedelta

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.dates import DateParseError, describe_day, parse_range, parse_when
from jarvis.errors import NeedsClarificationError, ToolExecutionError, ToolInputError
from jarvis.platforms.base import CalendarApp, CalendarEvent, PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec
from jarvis.tools.timers import parse_duration

DEFAULT_LENGTH = timedelta(hours=1)
MAX_LISTED = 40
_RELATIVE = {"today", "tomorrow", "this week", "next week", "this weekend"}


class EventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    when: str | None = Field(
        default=None, max_length=80, description='e.g. "tomorrow", "friday", "this week"'
    )


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)
    when: str = Field(min_length=1, max_length=80, description='e.g. "friday at 1pm"')
    duration: str | None = Field(default=None, max_length=40, description='e.g. "30 minutes"')
    calendar: str | None = Field(default=None, max_length=100)
    location: str | None = Field(default=None, max_length=200)


def time_range(start: datetime, end: datetime) -> str:
    """ "9:30 to 9:45 AM", "11:30 AM to 1:00 PM" (reads well when spoken)."""
    s, e = start.astimezone(), end.astimezone()
    if s.date() != e.date():
        return f"{s:%-I:%M %p} to {e:%a %-I:%M %p}"
    if f"{s:%p}" == f"{e:%p}":
        return f"{s:%-I:%M} to {e:%-I:%M %p}"
    return f"{s:%-I:%M %p} to {e:%-I:%M %p}"


def describe_event(event: CalendarEvent) -> str:
    when = "All day" if event.all_day else time_range(event.start, event.end)
    where = f", {event.location}" if event.location else ""
    return f"{when}: {event.title}{where} ({event.calendar})"


def _for(label: str) -> str:
    """ "today" -> "today"; "Fri 4 Dec" -> "on Fri 4 Dec"."""
    return label if label in _RELATIVE else f"on {label}"


def _day(event: CalendarEvent) -> date:
    return event.start.astimezone().date()


class CalendarTools:
    """Builds the ``calendar.*`` tool specs."""

    def __init__(
        self,
        platform: PlatformAdapter,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._platform = platform
        self._clock = clock

    def _app(self) -> CalendarApp:
        return self._platform.calendar()  # UnsupportedPlatformError off macOS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("calendar.events", "List calendar events for a day or a week.", EventsArgs,
                     RiskLevel.READ_ONLY, self._prepare_events, idempotent=True),
            ToolSpec("calendar.add", "Add an event to the calendar (asks first).", AddArgs,
                     RiskLevel.MEDIUM, self._prepare_add),
        ]  # fmt: skip

    def _prepare_events(self, args: EventsArgs) -> PreparedAction:
        app = self._app()
        today = self._clock().date()
        try:
            first, last, label = parse_range(args.when or "today", today)
        except DateParseError as exc:
            raise ToolInputError(str(exc)) from exc

        def run(cancel: threading.Event) -> ToolResult:
            start = datetime.combine(first, time(0, 0)).astimezone()
            end = datetime.combine(last + timedelta(days=1), time(0, 0)).astimezone()
            listing = app.events(start, end)
            events = listing.events[:MAX_LISTED]
            if not events:
                message = f"Nothing on your calendar {_for(label)}."
            else:
                lines = [f"Your calendar {_for(label)} ({len(listing.events)}):"]
                if first == last:
                    lines += [f"  {describe_event(e)}" for e in events]
                else:
                    for day in sorted({_day(e) for e in events}):
                        heading = describe_day(day, today)
                        lines.append(f"{heading[0].upper()}{heading[1:]}:")
                        lines += [f"  {describe_event(e)}" for e in events if _day(e) == day]
                message = "\n".join(lines)
            if len(listing.events) > MAX_LISTED:
                message += f"\n  … and {len(listing.events) - MAX_LISTED} more"
            if listing.notes:
                message += "\n" + "\n".join(f"Note: {n}" for n in listing.notes)
            return ToolResult(
                tool="calendar.events",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"count": len(listing.events), "from": str(first), "to": str(last)},
                untrusted=bool(events),  # titles and places come from invitations too
            )

        return PreparedAction(description=f"Read your calendar {_for(label)}", run=run)

    @staticmethod
    def _resolve_calendar(app: CalendarApp, wanted: str) -> str:
        names = app.calendars()
        by_key = {n.casefold(): n for n in names}
        key = wanted.casefold().removesuffix(" calendar").strip()
        if key in by_key:
            return by_key[key]
        close = difflib.get_close_matches(key, list(by_key), n=2, cutoff=0.75)
        if close:
            options = " or ".join(f"“{by_key[c]}”" for c in close)
            raise NeedsClarificationError(
                f"There's no calendar called “{wanted}”. Did you mean {options}?"
            )
        raise ToolExecutionError(
            f"There's no calendar called “{wanted}” that events can be added to.",
            hint="Your calendars: " + ", ".join(names) if names else None,
        )

    def _prepare_add(self, args: AddArgs) -> PreparedAction:
        app = self._app()
        now = self._clock()
        title = " ".join(args.title.split())
        try:
            when = parse_when(args.when, now)
        except DateParseError as exc:
            raise ToolInputError(str(exc), hint='e.g. "friday at 1pm"') from exc
        if when is None:
            raise ToolInputError(f"“{args.when}” isn't a day or time.", hint='e.g. "friday at 1pm"')
        length = parse_duration(args.duration) if args.duration else DEFAULT_LENGTH
        start = when.start()
        all_day = when.at is None
        end = start + timedelta(days=1) if all_day else start + length
        location = (args.location or "").strip()
        day = describe_day(when.day, now.date())
        span = "all day" if all_day else time_range(start, end)
        where = f" at {location}" if location else ""
        wanted_calendar = args.calendar

        def run(cancel: threading.Event) -> ToolResult:
            calendar = self._resolve_calendar(app, wanted_calendar) if wanted_calendar else None
            added = app.add_event(
                title, start, end, all_day=all_day, calendar=calendar, location=location
            )
            return ToolResult(
                tool="calendar.add",
                status=ToolStatus.SUCCESS,
                message=f"Added “{title}” to {added.calendar or 'your calendar'}: {day}, {span}.",
                data={"id": added.id, "start": start.isoformat(), "end": end.isoformat()},
            )

        return PreparedAction(
            description=f"Add “{title}” to your calendar: {day}, {span}{where}", run=run
        )
