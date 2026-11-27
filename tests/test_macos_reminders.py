"""The Reminders adapter with a fake osascript (Reminders is never driven)."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from typing import Any

import pytest

from jarvis.platforms.macos_reminders import (
    ADD,
    ALL_SCRIPTS,
    COMPLETE,
    LISTS,
    OPEN_REMINDERS,
    MacReminders,
)


class FakeOsascript:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.replies: dict[str, str] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, self.replies.get(argv[2], ""), "")


@pytest.fixture
def osascript() -> FakeOsascript:
    return FakeOsascript()


def test_lists_and_open_reminders(osascript: FakeOsascript) -> None:
    osascript.replies[LISTS] = "Reminders\nGroceries\n"
    osascript.replies[OPEN_REMINDERS] = (
        "x-apple-reminder://A\tBuy milk\tGroceries\t2026-11-28T18:00\tfalse\n"
        "x-apple-reminder://B\tPay rent\tReminders\t2026-12-01T00:00\ttrue\n"
        "x-apple-reminder://C\tCall Sam\tReminders\t\tfalse\n"
    )
    app = MacReminders(osascript)
    assert app.lists() == ["Reminders", "Groceries"]
    found = app.open_reminders("Groceries")
    assert osascript.calls[-1][3:] == ["Groceries"]
    assert [r.title for r in found] == ["Buy milk", "Pay rent", "Call Sam"]
    assert found[0].due == datetime(2026, 11, 28, 18, 0).astimezone()
    assert found[1].all_day and found[2].due is None
    app.open_reminders()
    assert osascript.calls[-1][3:] == [""]  # every list


def test_add_passes_text_and_date_as_arguments(osascript: FakeOsascript) -> None:
    osascript.replies[ADD] = "x-apple-reminder://D\tReminders\n"
    app = MacReminders(osascript)
    tricky = 'milk" & (do shell script "id") & "'
    due = datetime(2026, 12, 4, 13, 5).astimezone()
    added = app.add(tricky, None, due, notes="2%")
    assert osascript.calls[-1] == [
        "/usr/bin/osascript", "-e", ADD, tricky, "", "2%", "1", "0", "2026", "12", "4", "13", "5"
    ]  # fmt: skip
    assert added.id == "x-apple-reminder://D" and added.list_name == "Reminders"
    app.add("Pay rent", "Bills", datetime(2026, 12, 1).astimezone(), all_day=True)
    assert osascript.calls[-1][3:8] == ["Pay rent", "Bills", "", "1", "1"]
    app.add("Someday")
    assert osascript.calls[-1][3:] == ["Someday", "", "", "0", "0"]


def test_complete(osascript: FakeOsascript) -> None:
    MacReminders(osascript).complete("x-apple-reminder://A")
    assert osascript.calls[-1][2:] == [COMPLETE, "x-apple-reminder://A"]


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Reminders no Apple events: it neither starts nor changes it."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
def test_date_helpers_round_trip() -> None:
    """Runs only the helpers: no app is involved."""
    from jarvis.platforms.macos_reminders import DATES

    script = DATES + "\non run argv\n\treturn iso(mkdate(2027, 2, 28, 13, 5))\nend run"
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2027-02-28T13:05"
