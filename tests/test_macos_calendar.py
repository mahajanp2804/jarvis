"""The Calendar adapter with a fake osascript (Calendar is never driven)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from typing import Any

import pytest

from jarvis.platforms.macos_calendar import (
    ADD_EVENT,
    ALL_SCRIPTS,
    EVENTS,
    MacCalendar,
    parse_events,
)


def local(*parts: int) -> datetime:
    return datetime(*parts).astimezone()  # type: ignore[arg-type]


DAY = (local(2026, 11, 30), local(2026, 12, 1))  # Monday 30 November


def row(*values: str) -> str:
    return "\t".join(values) + "\n"


def test_single_repeating_and_all_day_events() -> None:
    text = (
        row("u1", "Dentist", "Home", "2026-11-30T15:00", "2026-11-30T16:00", "false", "Main St")
        + row(
            "u2",
            "Standup",
            "Work",
            "2025-01-06T09:30",
            "2025-01-06T09:45",
            "false",
            "",
            "FREQ=WEEKLY;BYDAY=MO,WE",
        )
        + row(
            "u3",
            "Mom's birthday",
            "Birthdays",
            "1960-11-30T00:00",
            "1960-12-01T00:00",
            "true",
            "",
            "FREQ=YEARLY",
        )
        + row("u4", "Yesterday", "Home", "2026-11-29T10:00", "2026-11-29T11:00", "false")
        + row(
            "u5",
            "Skipped standup",
            "Work",
            "2025-01-06T11:00",
            "2025-01-06T11:15",
            "false",
            "",
            "FREQ=WEEKLY",
            "2026-11-30T11:00,",
        )
    )
    listing = parse_events(text, *DAY)
    assert [(e.title, f"{e.start:%H:%M}", e.all_day) for e in listing.events] == [
        ("Mom's birthday", "00:00", True),  # all-day events first
        ("Standup", "09:30", False),
        ("Dentist", "15:00", False),
    ]
    assert listing.events[1].end == local(2026, 11, 30, 9, 45)
    assert listing.events[2].location == "Main St" and listing.notes == []


def test_what_could_be_missing_is_reported() -> None:
    text = row("#norepeat", "Work") + row(
        "u1", "Odd", "Home", "2026-01-01T09:00", "2026-01-01T10:00", "false", "", "FREQ=HOURLY"
    )
    listing = parse_events(text, *DAY)
    assert listing.events == []
    assert listing.notes == [
        "Repeating events in “Work” couldn't be read and may be missing.",
        "“Odd” repeats in a way Jarvis can't work out; check Calendar.",
    ]


def test_an_overnight_repeat_that_started_the_day_before_overlaps() -> None:
    night = row(
        "u1",
        "Night shift",
        "Work",
        "2026-11-01T22:00",
        "2026-11-02T06:00",
        "false",
        "",
        "FREQ=DAILY",
    )
    starts = [f"{e.start:%d %H:%M}" for e in parse_events(night, *DAY).events]
    assert starts == ["29 22:00", "30 22:00"]


class FakeOsascript:
    def __init__(self, reply: str = "") -> None:
        self.calls: list[list[str]] = []
        self.reply = reply

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, self.reply, "")


def test_dates_and_text_go_in_as_arguments() -> None:
    osascript = FakeOsascript()
    MacCalendar(osascript).events(*DAY)
    assert osascript.calls[-1][2] == EVENTS
    assert osascript.calls[-1][3:] == ["2026", "11", "30", "0", "0", "2026", "12", "1", "0", "0"]
    osascript.reply = "u9\tHome\n"
    tricky = 'Lunch" & (do shell script "id") & "'
    added = MacCalendar(osascript).add_event(
        tricky, local(2026, 12, 4, 13, 0), local(2026, 12, 4, 14, 0), location="Cafe"
    )
    assert osascript.calls[-1][2:] == [
        ADD_EVENT,
        tricky,
        "",
        "Cafe",
        "0",
        "2026",
        "12",
        "4",
        "13",
        "0",
        "2026",
        "12",
        "4",
        "14",
        "0",
    ]
    assert added.id == "u9" and added.calendar == "Home"


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Calendar no Apple events: it neither starts nor changes it."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
