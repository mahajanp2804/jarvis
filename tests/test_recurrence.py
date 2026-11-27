from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from jarvis.recurrence import UnsupportedRule, occurrences


def local(*parts: int) -> datetime:
    return datetime(*parts).astimezone()  # type: ignore[arg-type]


def days(found: list[datetime]) -> list[str]:
    return [f"{d:%a %d %b %H:%M}" for d in found]


WEEK = (local(2026, 11, 30), local(2026, 12, 7))  # Mon 30 Nov - Sun 6 Dec


def test_weekly_on_given_days_started_long_ago() -> None:
    standup = local(2025, 1, 6, 9, 30)  # a Monday
    found = occurrences("FREQ=WEEKLY;INTERVAL=1;BYDAY=MO,WE,FR", standup, *WEEK)
    assert days(found) == ["Mon 30 Nov 09:30", "Wed 02 Dec 09:30", "Fri 04 Dec 09:30"]


def test_every_other_week_and_excluded_dates() -> None:
    first = local(2026, 11, 2, 14, 0)  # Mon 2 Nov
    two_weeks = (local(2026, 11, 2), local(2026, 12, 15))
    assert days(occurrences("FREQ=WEEKLY;INTERVAL=2", first, *two_weeks)) == [
        "Mon 02 Nov 14:00", "Mon 16 Nov 14:00", "Mon 30 Nov 14:00", "Mon 14 Dec 14:00",
    ]  # fmt: skip
    skipped = occurrences(
        "FREQ=WEEKLY;INTERVAL=2", first, *two_weeks, excluded=frozenset({date(2026, 11, 16)})
    )
    assert len(skipped) == 3


def test_daily_with_count_and_until() -> None:
    first = local(2026, 11, 26, 8, 0)
    assert days(occurrences("FREQ=DAILY;COUNT=5", first, *WEEK)) == ["Mon 30 Nov 08:00"]
    assert len(occurrences("FREQ=DAILY;UNTIL=20261202", first, *WEEK)) == 3  # 30 Nov - 2 Dec
    assert occurrences("FREQ=DAILY;UNTIL=20261101T000000Z", first, *WEEK) == []


def test_monthly_rules() -> None:
    first = local(2026, 1, 13, 18, 0)  # second Tuesday of January
    december = (local(2026, 12, 1), local(2027, 1, 1))
    assert days(occurrences("FREQ=MONTHLY;BYDAY=2TU", first, *december)) == ["Tue 08 Dec 18:00"]
    assert days(occurrences("FREQ=MONTHLY;BYDAY=-1FR", first, *december)) == ["Fri 25 Dec 18:00"]
    assert days(occurrences("FREQ=MONTHLY;BYMONTHDAY=1,-1", first, *december)) == [
        "Tue 01 Dec 18:00", "Thu 31 Dec 18:00",
    ]  # fmt: skip
    rent = local(2026, 1, 31, 9, 0)  # months without a 31st are skipped
    assert len(occurrences("FREQ=MONTHLY", rent, local(2026, 2, 1), local(2026, 6, 1))) == 2


def test_yearly_birthday() -> None:
    birthday = local(1990, 12, 3)
    assert days(occurrences("FREQ=YEARLY", birthday, *WEEK)) == ["Thu 03 Dec 00:00"]
    leap = local(2024, 2, 29)
    assert occurrences("FREQ=YEARLY", leap, local(2026, 1, 1), local(2027, 1, 1)) == []


def test_unknown_parts_are_not_guessed() -> None:
    with pytest.raises(UnsupportedRule):
        occurrences("FREQ=HOURLY", local(2026, 1, 1), *WEEK)
    with pytest.raises(UnsupportedRule):
        occurrences("FREQ=MONTHLY;BYSETPOS=-1;BYDAY=MO,TU", local(2026, 1, 1), *WEEK)


def test_long_daily_series_is_fast() -> None:
    first = local(1996, 1, 1, 7, 0)
    window = (local(2026, 11, 30), local(2026, 11, 30) + timedelta(days=1))
    assert days(occurrences("FREQ=DAILY", first, *window)) == ["Mon 30 Nov 07:00"]
