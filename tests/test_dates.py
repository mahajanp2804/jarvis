from __future__ import annotations

from datetime import date, datetime, time

import pytest

from jarvis.dates import (
    DateParseError,
    describe_when,
    parse_day,
    parse_range,
    parse_time,
    parse_when,
)

TODAY = date(2026, 11, 28)  # a Saturday


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("friday", date(2026, 12, 4)), ("on fri", date(2026, 12, 4)),
        ("saturday", TODAY), ("next saturday", date(2026, 12, 5)),
        ("tomorrow", date(2026, 11, 29)), ("the day after tomorrow", date(2026, 11, 30)),
        ("tonight", TODAY), ("Dec 3rd", date(2026, 12, 3)), ("3 December", date(2026, 12, 3)),
        ("the 3rd of december", date(2026, 12, 3)), ("the 5th", date(2026, 12, 5)),
        ("the 30th", date(2026, 11, 30)), ("Jan 5", date(2027, 1, 5)),
        ("in 3 days", date(2026, 12, 1)), ("in a week", date(2026, 12, 5)),
        ("2026-12-24", date(2026, 12, 24)), ("buy milk", None),
    ],
)  # fmt: skip
def test_parse_day(text: str, expected: date | None) -> None:
    assert parse_day(text, TODAY) == expected


@pytest.mark.parametrize("text", ["Feb 30", "2026-13-01", "the 31st"])
def test_days_that_do_not_exist(text: str) -> None:
    with pytest.raises(DateParseError):
        parse_day(text, date(2026, 11, 1) if "31st" in text else TODAY)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("at 1", time(13)), ("at 9", time(9)), ("at 12", time(12)), ("at 07:30", time(7, 30)),
        ("7:30 pm", time(19, 30)), ("7pm", time(19)), ("12 am", time(0)), ("18:45", time(18, 45)),
        ("noon", time(12)), ("dec 3", None), ("the 3rd", None), ("in 3 days", None),
        ("2026-12-24", None),
    ],
)  # fmt: skip
def test_parse_time(text: str, expected: time | None) -> None:
    assert parse_time(text) == expected


def test_bad_times() -> None:
    for text in ("at 13 pm", "at 25:00"):
        with pytest.raises(DateParseError):
            parse_time(text)


def test_parse_when() -> None:
    now = datetime(2026, 11, 28, 16, 0).astimezone()
    assert describe_when(parse_when("lunch with sam friday at 1", now), TODAY) == (
        "Fri 4 Dec at 1:00 PM"
    )
    assert describe_when(parse_when("dentist tomorrow 9:30am", now), TODAY) == (
        "tomorrow at 9:30 AM"
    )
    assert parse_when("at 3pm", now).day == date(2026, 11, 29)
    assert parse_when("at 5pm", now).day == TODAY
    whole_day = parse_when("Dec 3", now)
    assert whole_day is not None and whole_day.at is None
    assert parse_when("call mom", now) is None
    assert describe_when(parse_when("tomorrow morning", now), TODAY) == "tomorrow at 9:00 AM"
    assert describe_when(parse_when("friday afternoon", now), TODAY) == "Fri 4 Dec at 2:00 PM"
    assert describe_when(parse_when("tonight", now), TODAY) == "today at 8:00 PM"
    assert describe_when(parse_when("tonight at 9:30", now), TODAY) == "today at 9:30 PM"
    assert describe_when(parse_when("in 20 minutes", now), TODAY) == "today at 4:20 PM"
    assert describe_when(parse_when("in half an hour", now), TODAY) == "today at 4:30 PM"
    assert describe_when(parse_when("in 9 hours", now), TODAY) == "tomorrow at 1:00 AM"


def test_parse_range() -> None:
    assert parse_range("what's on my calendar", TODAY) == (TODAY, TODAY, "today")
    assert parse_range("this week", TODAY) == (TODAY, date(2026, 11, 29), "this week")
    assert parse_range("next week", TODAY) == (date(2026, 11, 30), date(2026, 12, 6), "next week")
    assert parse_range("this weekend", TODAY) == (TODAY, date(2026, 11, 29), "this weekend")
    assert parse_range("on monday", TODAY) == (date(2026, 11, 30), date(2026, 11, 30), "Mon 30 Nov")
