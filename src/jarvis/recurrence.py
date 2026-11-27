"""Expand a repeating event's rule (iCalendar RRULE) into its dates in a window.

Calendar's AppleScript returns a repeating event once, at its first date, with
its rule as text ("FREQ=WEEKLY;INTERVAL=1;BYDAY=MO,WE"). This works out which
occurrences fall in the days asked about, so "what's on tomorrow?" includes a
weekly meeting that started months ago.

Supported: FREQ DAILY, WEEKLY, MONTHLY, YEARLY; INTERVAL; COUNT; UNTIL; BYDAY
(weekdays, and "2TU" / "-1FR" for monthly); BYMONTHDAY. Anything else raises
:class:`UnsupportedRule`, so the caller can say the event repeats instead of
guessing its dates. Times keep the first occurrence's local wall-clock time,
across daylight-saving changes.
"""

from __future__ import annotations

import calendar
import re
from datetime import UTC, date, datetime, timedelta

_DAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
_BYDAY = re.compile(r"^([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)$")
_IGNORED = {"WKST"}
MAX_STEPS = 10_000  # periods examined; enough for decades of a daily rule


class UnsupportedRule(ValueError):
    """A rule part this expander doesn't handle."""


def _parse_until(value: str) -> datetime:
    if re.fullmatch(r"\d{8}", value):
        day = datetime.strptime(value, "%Y%m%d")
        return day.replace(hour=23, minute=59, second=59).astimezone()
    if re.fullmatch(r"\d{8}T\d{6}Z", value):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    if re.fullmatch(r"\d{8}T\d{6}", value):
        return datetime.strptime(value, "%Y%m%dT%H%M%S").astimezone()
    raise UnsupportedRule(f"UNTIL={value}")


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date | None:
    days = [
        date(year, month, d)
        for d in range(1, calendar.monthrange(year, month)[1] + 1)
        if date(year, month, d).weekday() == weekday
    ]
    try:
        return days[nth - 1] if nth > 0 else days[nth]
    except IndexError:
        return None


def _month_days(year: int, month: int, parts: dict[str, str], first: date) -> list[date]:
    last_day = calendar.monthrange(year, month)[1]
    if "BYMONTHDAY" in parts:
        found = []
        for value in parts["BYMONTHDAY"].split(","):
            day = int(value)
            day = last_day + day + 1 if day < 0 else day
            if 1 <= day <= last_day:
                found.append(date(year, month, day))
        return sorted(found)
    if "BYDAY" in parts:
        found = []
        for value in parts["BYDAY"].split(","):
            match = _BYDAY.match(value)
            if not match:
                raise UnsupportedRule(f"BYDAY={value}")
            weekday = _DAYS[match.group(2)]
            if match.group(1):
                nth = _nth_weekday(year, month, weekday, int(match.group(1)))
                if nth:
                    found.append(nth)
            else:  # every such weekday in the month
                found += [
                    date(year, month, d)
                    for d in range(1, last_day + 1)
                    if date(year, month, d).weekday() == weekday
                ]
        return sorted(found)
    return [date(year, month, first.day)] if first.day <= last_day else []


def occurrences(
    rule: str,
    first: datetime,
    window_start: datetime,
    window_end: datetime,
    excluded: frozenset[date] = frozenset(),
) -> list[datetime]:
    """Start times of the occurrences that start before ``window_end`` and whose
    start is at or after ``window_start`` (callers widen the window by the
    event's length to catch ones that overlap it)."""
    parts = dict(
        p.split("=", 1) for p in rule.upper().removeprefix("RRULE:").split(";") if "=" in p
    )
    unknown = set(parts) - {"FREQ", "INTERVAL", "COUNT", "UNTIL", "BYDAY", "BYMONTHDAY"} - _IGNORED
    if unknown:
        raise UnsupportedRule(", ".join(sorted(unknown)))
    freq = parts.get("FREQ", "")
    interval = int(parts.get("INTERVAL", "1"))
    if interval < 1:
        raise UnsupportedRule(f"INTERVAL={interval}")
    count = int(parts["COUNT"]) if "COUNT" in parts else None
    until = _parse_until(parts["UNTIL"]) if "UNTIL" in parts else None
    local_first = first.astimezone()
    wall = local_first.replace(tzinfo=None)  # wall-clock time to keep

    def at(day: date) -> datetime:
        return datetime.combine(day, wall.time()).astimezone()

    def days_in_period(step: int) -> list[date]:
        if freq == "DAILY":
            return [wall.date() + timedelta(days=step * interval)]
        if freq == "WEEKLY":
            week = wall.date() - timedelta(days=wall.weekday()) + timedelta(weeks=step * interval)
            if "BYDAY" not in parts:
                return [wall.date() + timedelta(weeks=step * interval)]
            weekdays = []
            for value in parts["BYDAY"].split(","):
                match = _BYDAY.match(value)
                if not match or match.group(1):
                    raise UnsupportedRule(f"BYDAY={value}")
                weekdays.append(_DAYS[match.group(2)])
            return [week + timedelta(days=d) for d in sorted(set(weekdays))]
        if freq == "MONTHLY":
            index = wall.month - 1 + step * interval
            return _month_days(wall.year + index // 12, index % 12 + 1, parts, wall.date())
        if freq == "YEARLY":
            if "BYDAY" in parts or "BYMONTHDAY" in parts:
                raise UnsupportedRule("YEARLY with BYDAY or BYMONTHDAY")
            year = wall.year + step * interval
            last = calendar.monthrange(year, wall.month)[1]
            return [date(year, wall.month, wall.day)] if wall.day <= last else []
        raise UnsupportedRule(f"FREQ={freq or 'missing'}")

    # Without COUNT, nothing before the window matters: start just before it.
    first_step = 0
    gap = (window_start.astimezone().date() - wall.date()).days
    if count is None and gap > 0:
        period = {"DAILY": 1, "WEEKLY": 7, "MONTHLY": 31, "YEARLY": 366}.get(freq, 1)
        first_step = max(0, gap // (period * interval) - 1)

    found: list[datetime] = []
    produced = 0
    for step in range(first_step, first_step + MAX_STEPS):
        for day in days_in_period(step):
            if day < wall.date():
                continue  # before the first occurrence
            start = at(day)
            if until is not None and start > until:
                return found
            if count is not None and produced >= count:
                return found
            produced += 1
            if start >= window_end:
                return found
            if start >= window_start and day not in excluded:
                found.append(start)
    return found
