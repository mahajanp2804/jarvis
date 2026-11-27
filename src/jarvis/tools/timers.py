"""Timer and alarm tools: start a timer, set an alarm, list and cancel them.

* All LOW risk, like Siri's: nothing on the Mac changes until one rings.
* An alarm with a label is also a reminder: "remind me to buy milk at 6 pm".
* They ring while the Jarvis service runs (a dialog, a sound and, with speech
  on, a spoken line). If the service isn't running when one is set, the reply
  says so; a timer that came due while it was stopped rings when it starts.
* Durations and times are parsed here, not by the model: "10 minutes",
  "1h30m", "half an hour"; "7 am", "18:45", "7:30 pm tomorrow", "noon".
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import NeedsClarificationError, ToolInputError
from jarvis.timers.store import Timer, TimerStore
from jarvis.tools.base import PreparedAction, ToolSpec

MAX_TIMER = timedelta(hours=24)

_UNITS = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
}  # fmt: skip
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?|an?|half an?)\s*([a-z]+)")


def parse_duration(text: str) -> timedelta:
    """ "10 minutes", "1h30m", "1.5 hours", "half an hour", "90 sec"; a bare number is minutes."""
    words = text.strip().lower().replace(",", " ").replace(" and ", " ")
    if re.fullmatch(r"\d+(?:\.\d+)?", words):
        words += " minutes"
    words = re.sub(r"(\d)([a-z])", r"\1 \2", words)  # 1h30m -> 1 h30 m -> handled below
    words = re.sub(r"([a-z])(\d)", r"\1 \2", words)
    total = 0.0
    found = False
    for amount, unit in _AMOUNT.findall(words):
        if unit not in _UNITS:
            continue
        value = (
            0.5 if amount.startswith("half") else 1.0 if amount in ("a", "an") else float(amount)
        )
        total += value * _UNITS[unit]
        found = True
    if not found or total < 1:
        raise ToolInputError(f"“{text}” isn't a length of time.", hint="e.g. “10 minutes”")
    if total > MAX_TIMER.total_seconds():
        raise ToolInputError("Timers can be at most 24 hours.", hint="Set an alarm instead.")
    return timedelta(seconds=round(total))


_DURATION_WORD = re.compile(r"\b(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)\b")
_LATER_DAY = re.compile(
    r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|\b\d{1,2}(?:st|nd|rd|th)\b"
    r"|\b(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\bnext week\b"
)


def mentions_later_day(text: str) -> bool:
    """ "on friday", "the 3rd", "in december": beyond today and tomorrow."""
    return bool(_LATER_DAY.search(text.lower()))


_CLOCK = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d:])", re.IGNORECASE)


def parse_alarm_time(text: str, now: datetime) -> datetime:
    """The next "7 am", "18:45", "7:30 pm tomorrow" or "noon" (local), or "in 20 minutes"."""
    words = text.strip().lower()
    if _DURATION_WORD.search(words):  # "in 20 minutes": that long from now
        return now.astimezone() + parse_duration(words.removeprefix("in ").strip())
    if mentions_later_day(words):
        raise ToolInputError(
            "Alarms and reminders can be set for today or tomorrow only.",
            hint="Use the Reminders or Calendar app for later dates.",
        )
    either = False
    if "noon" in words or "midday" in words:
        hour, minute = 12, 0
    elif "midnight" in words:
        hour, minute = 0, 0
    else:
        match = _CLOCK.search(words)
        if not match:
            raise ToolInputError(f"“{text}” isn't a time of day.", hint="e.g. “7:30 am”")
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        meridiem = (match.group(3) or "").replace(".", "")
        # "at 7" without am/pm: whichever of 7:00 and 19:00 comes next ("07:30" is 24-hour).
        either = not meridiem and 1 <= hour <= 12 and not match.group(1).startswith("0")
        if meridiem and not 1 <= hour <= 12:
            raise ToolInputError(f"“{text}” isn't a time of day.", hint="e.g. “7:30 am”")
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            raise ToolInputError(f"“{text}” isn't a time of day.", hint="e.g. “18:45”")
    local_now = now.astimezone()
    start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    if "tomorrow" in words:
        start += timedelta(days=1)
    hours = [hour % 12, hour % 12 + 12] if either else [hour]
    candidates = [
        start + timedelta(days=day, hours=h, minutes=minute) for day in (0, 1) for h in hours
    ]
    # The first one still ahead; "tomorrow" without am/pm picks the morning.
    return next(c for c in candidates if c > local_now)


def _clock_time(moment: datetime) -> str:
    return f"{moment.astimezone():%-I:%M %p}"


def describe_when(moment: datetime, now: datetime) -> str:
    local, today = moment.astimezone(), now.astimezone().date()
    if local.date() == today:
        return _clock_time(local)
    if local.date() == today + timedelta(days=1):
        return f"{_clock_time(local)} tomorrow"
    return f"{_clock_time(local)} on {local:%a %-d %b}"


def describe_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours} hour{'s' * (hours != 1)}"] if hours else []
    if minutes:
        parts.append(f"{minutes} minute{'s' * (minutes != 1)}")
    if secs and not hours:
        parts.append(f"{secs} second{'s' * (secs != 1)}")
    return " ".join(parts) or "0 seconds"


def timer_name(timer: Timer) -> str:
    return f"the “{timer.label}” {timer.kind}" if timer.label else f"the {timer.kind}"


def describe_timer(timer: Timer, now: datetime) -> str:
    name = timer_name(timer).removeprefix("the ")
    name = name[0].upper() + name[1:]
    if timer.kind == "timer":
        left = describe_duration((timer.due_at - now).total_seconds())
        return f"{name}: {left} left (rings at {describe_when(timer.due_at, now)})"
    return f"{name}: {describe_when(timer.due_at, now)}"


class StartArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration: str = Field(min_length=1, max_length=60, description='e.g. "10 minutes"')
    label: str | None = Field(default=None, max_length=100, description="what it's for")


class AlarmArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: str = Field(min_length=1, max_length=60, description='e.g. "7:30 am tomorrow"')
    label: str | None = Field(default=None, max_length=100, description="what it's for")


class ListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CancelArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    which: str = Field(min_length=1, max_length=100, description='its label or number, or "all"')


class TimerTools:
    """Builds the ``timers.*`` tool specs.

    ``changed`` is called after a timer is added or cancelled, so whatever rings
    them can re-read the database. It returns False when nothing is running to
    ring them (the reply then says so).
    """

    def __init__(
        self,
        store: TimerStore,
        changed: Callable[[], bool] = lambda: False,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._store = store
        self._changed = changed
        self._clock = clock

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("timers.start", "Start a timer.", StartArgs, RiskLevel.LOW,
                     self._prepare_start),
            ToolSpec("timers.alarm", "Set an alarm that rings on this Mac.", AlarmArgs,
                     RiskLevel.LOW, self._prepare_alarm),
            ToolSpec("timers.list", "List the pending timers and alarms.", ListArgs,
                     RiskLevel.READ_ONLY, self._prepare_list, idempotent=True),
            ToolSpec("timers.cancel", "Cancel a timer or alarm.", CancelArgs, RiskLevel.LOW,
                     self._prepare_cancel),
        ]  # fmt: skip

    def _ringing_note(self) -> str:
        if self._changed():
            return ""
        return (
            "\nThe Jarvis service isn't running, so it won't ring until you start it "
            "(jarvis service start)."
        )

    def _added(self, tool: str, timer: Timer, message: str) -> ToolResult:
        return ToolResult(
            tool=tool,
            status=ToolStatus.SUCCESS,
            message=message + self._ringing_note(),
            data={"id": timer.id, "due_at": timer.due_at.isoformat()},
        )

    def _prepare_start(self, args: StartArgs) -> PreparedAction:
        length = parse_duration(args.duration)
        label = (args.label or "").strip()
        named = f" “{label}”" if label else ""
        spoken = describe_duration(length.total_seconds())

        def run(cancel: threading.Event) -> ToolResult:
            now = self._clock()
            timer = self._store.add(
                "timer", now + length, label, duration_seconds=int(length.total_seconds())
            )
            when = describe_when(timer.due_at, now)
            return self._added("timers.start", timer, f"Timer{named} set for {spoken} ({when}).")

        return PreparedAction(description=f"Start a {spoken} timer{named}", run=run)

    def _prepare_alarm(self, args: AlarmArgs) -> PreparedAction:
        due = parse_alarm_time(args.time, self._clock())
        label = (args.label or "").strip()
        named = f" “{label}”" if label else ""
        kind_word = "Reminder" if label else "Alarm"

        def run(cancel: threading.Event) -> ToolResult:
            now = self._clock()
            timer = self._store.add("alarm", due, label)
            when = describe_when(due, now)
            return self._added("timers.alarm", timer, f"{kind_word}{named} set for {when}.")

        when = describe_when(due, self._clock())
        return PreparedAction(description=f"Set {kind_word.lower()}{named} for {when}", run=run)

    def _prepare_list(self, args: ListArgs) -> PreparedAction:
        def run(cancel: threading.Event) -> ToolResult:
            now = self._clock()
            pending = self._store.pending()
            if not pending:
                message = "No timers or alarms."
            else:
                lines = [f"  {t.id}. {describe_timer(t, now)}" for t in pending]
                message = f"Timers and alarms ({len(pending)}):\n" + "\n".join(lines)
            return ToolResult(
                tool="timers.list",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"ids": [t.id for t in pending]},
            )

        return PreparedAction(description="List timers and alarms", run=run)

    def _select(self, which: str) -> list[Timer]:
        pending = self._store.pending()
        key = which.strip().casefold().removeprefix("the ").removesuffix(" timer")
        key = key.removesuffix(" alarm").strip()
        if key in {"all", "everything", "all timers", "all alarms"}:
            return pending
        if key.isdigit():
            return [t for t in pending if t.id == int(key)]
        if key in {"timer", "alarm"}:  # "cancel the timer": every timer (asks if several)
            return [t for t in pending if t.kind == key]
        if key in {"", "it", "that"}:
            return pending
        return [t for t in pending if key in t.label.casefold()]

    def _prepare_cancel(self, args: CancelArgs) -> PreparedAction:
        chosen = self._select(args.which)
        everything = args.which.strip().casefold() in {"all", "everything"}
        if not chosen:
            pending = self._store.pending()
            raise ToolInputError(
                f"There's no timer or alarm matching “{args.which}”.",
                hint="Pending: " + "; ".join(f"{t.id}. {t.label or t.kind}" for t in pending)
                if pending
                else "Nothing is pending.",
            )
        if len(chosen) > 1 and not everything:
            options = ", ".join(f"{t.id}. {t.label or t.kind}" for t in chosen)
            raise NeedsClarificationError(f"Which one: {options}?")
        names = [timer_name(t) for t in chosen]

        def run(cancel: threading.Event) -> ToolResult:
            ended = [t for t in chosen if self._store.cancel(t.id)]
            self._changed()
            message = (
                f"Cancelled {len(ended)} timers and alarms."
                if len(ended) > 1
                else f"Cancelled {names[0]}."
                if ended
                else "It had already rung."
            )
            return ToolResult(
                tool="timers.cancel",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"cancelled": [t.id for t in ended]},
            )

        description = (
            f"Cancel {len(chosen)} timers and alarms" if len(chosen) > 1 else (f"Cancel {names[0]}")
        )
        return PreparedAction(description=description, run=run)
