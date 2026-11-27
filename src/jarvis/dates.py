"""Days, times and date ranges in requests, parsed by Jarvis rather than the model.

Used by the Reminders and Calendar tools: "friday at 1", "next tuesday",
"Dec 3rd", "the 3rd", "in 3 days", "this week". The model passes the words
through, and every confirmation shows the exact date that was understood
("Fri 4 Dec, 1:00 PM"), so a misreading is visible before anything is saved.

* A weekday is the next one, today included: "friday" on a Friday is today.
  "next friday" is the first one after today.
* A time without am/pm on a given day uses working hours: 7-11 are morning,
  12-6 afternoon ("lunch at 1" is 1 PM). "07:30" and "18:45" are as written.
* Parts of the day: morning 9 AM, afternoon 2 PM, evening 6 PM, tonight 8 PM.
* A month and day without a year is the next one: "Jan 5" in December is
  next year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

_WEEKDAYS = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1, "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3, "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}  # fmt: skip
_MONTHS = {
    name: number
    for number, names in enumerate(
        [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
         ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
         ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
         ("dec", "december")],
        start=1,
    )
    for name in names
}  # fmt: skip
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_WEEKDAY = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_ORDINAL = r"(\d{1,2})(?:st|nd|rd|th)?"

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+{_ORDINAL}\b", re.IGNORECASE)
_DAY_MONTH = re.compile(rf"\b{_ORDINAL}(?:\s+of)?\s+({_MONTH})\b", re.IGNORECASE)
_THE_NTH = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_IN_DAYS = re.compile(r"\bin\s+(\d{1,3}|a|one|two|three)\s+(day|days|week|weeks)\b", re.I)
_WEEKDAY_RE = re.compile(rf"\b(next\s+|this\s+|on\s+)?({_WEEKDAY})\b", re.IGNORECASE)
_CLOCK = re.compile(
    r"(?<![\d:-])(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d:])", re.IGNORECASE
)
_AT_CLOCK = re.compile(
    r"\b(?:at|by|from)\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\d:])", re.I
)
_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3}
# Parts of the day without a clock time, as phones read them.
_PARTS_OF_DAY = {
    "morning": time(9),
    "afternoon": time(14),
    "evening": time(18),
    "tonight": time(20),
}
_PART_OF_DAY = re.compile(r"\b(morning|afternoon|evening|tonight)\b", re.IGNORECASE)
_IN_TIME = re.compile(
    r"\bin\s+(\d{1,3}|an?|one|two|three|half an?)\s*(minutes?|mins?|hours?|hrs?)\b", re.I
)


class DateParseError(ValueError):
    """The text names a day or time that doesn't exist ("Feb 30", "25:00")."""


def _next_month_day(today: date, month: int, day: int) -> date:
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    raise DateParseError(f"There's no {day} of month {month}.")


def parse_day(text: str, today: date) -> date | None:
    """The day ``text`` names, or None if it names none."""
    words = text.lower()
    if match := _ISO.search(words):
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError as exc:
            raise DateParseError(f"{match.group(0)} isn't a date.") from exc
    if match := _MONTH_DAY.search(words):
        return _next_month_day(today, _MONTHS[match.group(1).lower()], int(match.group(2)))
    if match := _DAY_MONTH.search(words):
        return _next_month_day(today, _MONTHS[match.group(2).lower()], int(match.group(1)))
    if "day after tomorrow" in words:
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", words):
        return today + timedelta(days=1)
    if re.search(r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", words):
        return today
    if match := _IN_DAYS.search(words):
        amount = match.group(1)
        count = _WORD_NUMBERS.get(amount) or int(amount)
        return today + timedelta(days=count * (7 if match.group(2).startswith("week") else 1))
    if match := _WEEKDAY_RE.search(words):
        ahead = (_WEEKDAYS[match.group(2).lower()] - today.weekday()) % 7
        if (match.group(1) or "").strip() == "next" and ahead == 0:
            ahead = 7
        return today + timedelta(days=ahead)
    if match := _THE_NTH.search(words):
        day = int(match.group(1))
        month = today.month if day >= today.day else today.month % 12 + 1
        year = today.year + (1 if month < today.month else 0)
        try:
            return date(year, month, day)
        except ValueError as exc:
            raise DateParseError(f"That month has no {day}.") from exc
    return None


def parse_time(text: str, *, working_hours: bool = True) -> time | None:
    """The time of day ``text`` names ("at 1", "7:30 pm", "noon"), or None."""
    words = text.lower()
    if re.search(r"\b(?:noon|midday)\b", words):
        return time(12, 0)
    if re.search(r"\bmidnight\b", words):
        return time(0, 0)
    match = _AT_CLOCK.search(words)
    if match is None and (part := _PART_OF_DAY.search(words)):
        return _PARTS_OF_DAY[part.group(1)]
    if match is None:
        # "7pm" or "18:45"; a bare number is a day or a count, not a time
        match = next((m for m in _CLOCK.finditer(words) if m.group(3) or m.group(2)), None)
    if match is None:
        return None
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    meridiem = (match.group(3) or "").replace(".", "")
    if meridiem:
        if not 1 <= hour <= 12:
            raise DateParseError(f"{match.group(0).strip()} isn't a time.")
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    elif re.search(r"\b(?:tonight|evening|night)\b", words) and 1 <= hour < 12:
        hour += 12  # "tonight at 9:30" is 9:30 PM
    elif working_hours and 1 <= hour <= 6 and not match.group(1).startswith("0"):
        hour += 12  # "at 1" is 1 PM
    if hour > 23 or minute > 59:
        raise DateParseError(f"{match.group(0).strip()} isn't a time.")
    return time(hour, minute)


@dataclass(frozen=True)
class When:
    day: date
    at: time | None  # None: the whole day

    def start(self) -> datetime:
        return datetime.combine(self.day, self.at or time(0, 0)).astimezone()


def parse_when(text: str, now: datetime) -> When | None:
    """A day and maybe a time: "friday at 1", "tomorrow", "at 3pm" (today), "in 20 minutes".
    None if neither."""
    local = now.astimezone()
    if match := _IN_TIME.search(text):
        amount = match.group(1).lower()
        count = 0.5 if amount.startswith("half") else _WORD_NUMBERS.get(amount) or int(amount)
        minutes = count * (60 if match.group(2).lower().startswith("h") else 1)
        moment = (local + timedelta(minutes=minutes)).replace(second=0, microsecond=0)
        return When(moment.date(), moment.time())
    day = parse_day(text, local.date())
    at = parse_time(text)
    if day is None and at is None:
        return None
    if day is None and at is not None:
        today = local.date()
        later_today = datetime.combine(today, at, tzinfo=local.tzinfo) > local
        day = today if later_today else today + timedelta(days=1)  # "at 3pm" at 4 pm: tomorrow
    return When(day or local.date(), at)


def parse_range(text: str, today: date) -> tuple[date, date, str]:
    """Days to look at: (first, last, label) for "today", "this week", "next week", a day."""
    words = text.lower()
    monday = today - timedelta(days=today.weekday())
    if re.search(r"\bnext week\b", words):
        start = monday + timedelta(days=7)
        return start, start + timedelta(days=6), "next week"
    if re.search(r"\b(?:this|the) week\b|\bweek\b", words):
        return today, monday + timedelta(days=6), "this week"
    if re.search(r"\b(?:this )?weekend\b", words):
        saturday = monday + timedelta(days=5)
        start = max(saturday, today)
        return start, saturday + timedelta(days=1), "this weekend"
    day = parse_day(text, today) or today
    return day, day, describe_day(day, today)


def describe_day(day: date, today: date) -> str:
    if day == today:
        return "today"
    if day == today + timedelta(days=1):
        return "tomorrow"
    return f"{day:%a %-d %b}" + (f" {day.year}" if day.year != today.year else "")


def describe_when(when: When, today: date) -> str:
    """ "tomorrow at 1:00 PM", "Fri 4 Dec" (whole day)."""
    day = describe_day(when.day, today)
    return f"{day} at {when.at:%-I:%M %p}" if when.at else day
