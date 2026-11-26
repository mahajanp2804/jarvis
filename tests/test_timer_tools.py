from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from jarvis.api.schemas import ToolResult
from jarvis.errors import NeedsClarificationError, ToolInputError
from jarvis.timers.store import TimerStore
from jarvis.tools.base import PreparedAction
from jarvis.tools.timers import TimerTools, parse_alarm_time, parse_duration

NOW = datetime(2026, 11, 26, 9, 0).astimezone()  # Thursday, local time


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.now = NOW
        self.store = TimerStore(tmp_path / "timers.sqlite3", clock=lambda: self.now)
        self.ringer_running = True
        self.changes = 0
        self.tools = TimerTools(self.store, changed=self._changed, clock=lambda: self.now)
        self.specs = {spec.name: spec for spec in self.tools.specs()}

    def _changed(self) -> bool:
        self.changes += 1
        return self.ringer_running

    def prepare(self, tool: str, **args: Any) -> PreparedAction:
        spec = self.specs[tool]
        return spec.prepare(spec.args_model(**args))

    def run(self, tool: str, **args: Any) -> ToolResult:
        return self.prepare(tool, **args).run(threading.Event())


@pytest.fixture
def jarvis(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("10 minutes", 600), ("1h30m", 5400), ("1.5 hours", 5400), ("half an hour", 1800),
     ("90 sec", 90), ("an hour", 3600), ("5", 300), ("2 hours and 15 minutes", 8100)],
)  # fmt: skip
def test_parse_duration(text: str, seconds: int) -> None:
    assert parse_duration(text) == timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["soon", "0 minutes", "25 hours", ""])
def test_bad_durations(text: str) -> None:
    with pytest.raises(ToolInputError):
        parse_duration(text)


@pytest.mark.parametrize(
    ("hour", "text", "expected"),
    [
        (9, "7 am", "Fri 07:00"), (9, "at 7", "Thu 19:00"), (6, "at 7", "Thu 07:00"),
        (20, "at 7", "Fri 07:00"), (9, "07:30", "Fri 07:30"), (9, "18:45", "Thu 18:45"),
        (9, "7:30 pm tomorrow", "Fri 19:30"), (9, "7 tomorrow", "Fri 07:00"),
        (9, "noon", "Thu 12:00"), (9, "12 am", "Fri 00:00"), (9, "in 20 minutes", "Thu 09:20"),
    ],
)  # fmt: skip
def test_parse_alarm_time(hour: int, text: str, expected: str) -> None:
    now = NOW.replace(hour=hour)
    assert f"{parse_alarm_time(text, now):%a %H:%M}" == expected


@pytest.mark.parametrize("text", ["friday at 7", "7 am on the 3rd", "25:00", "13 pm", "soon"])
def test_bad_alarm_times(text: str) -> None:
    with pytest.raises(ToolInputError):
        parse_alarm_time(text, NOW)


def test_start_list_and_cancel(jarvis: Harness) -> None:
    started = jarvis.run("timers.start", duration="10 minutes", label="tea")
    assert started.message == "Timer “tea” set for 10 minutes (9:10 AM)."
    assert jarvis.prepare("timers.start", duration="1h").description == "Start a 1 hour timer"
    jarvis.run("timers.alarm", time="7 am", label="")
    listing = jarvis.run("timers.list").message
    assert listing == (
        "Timers and alarms (2):\n"
        "  1. “tea” timer: 10 minutes left (rings at 9:10 AM)\n"
        "  2. Alarm: 7:00 AM tomorrow"
    )
    jarvis.now = NOW + timedelta(minutes=4, seconds=30)
    assert "5 minutes 30 seconds left" in jarvis.run("timers.list").message
    assert jarvis.run("timers.cancel", which="the tea timer").message == (
        "Cancelled the “tea” timer."
    )
    assert jarvis.run("timers.cancel", which="alarm").message == "Cancelled the alarm."
    assert jarvis.run("timers.list").message == "No timers or alarms."
    assert jarvis.changes == 4  # two adds, two cancels: the ringer re-reads each time


def test_reminders_are_labelled_alarms(jarvis: Harness) -> None:
    result = jarvis.run("timers.alarm", time="6 pm", label="buy milk")
    assert result.message == "Reminder “buy milk” set for 6:00 PM."
    assert jarvis.store.pending()[0].kind == "alarm"


def test_the_reply_says_when_nothing_will_ring(jarvis: Harness) -> None:
    jarvis.ringer_running = False
    message = jarvis.run("timers.start", duration="5 minutes").message
    assert message.endswith("it won't ring until you start it (jarvis service start).")


def test_cancel_needs_a_clear_choice(jarvis: Harness) -> None:
    with pytest.raises(ToolInputError, match="no timer or alarm matching") as empty:
        jarvis.prepare("timers.cancel", which="timer")
    assert empty.value.hint == "Nothing is pending."
    jarvis.run("timers.start", duration="5 minutes", label="eggs")
    jarvis.run("timers.start", duration="9 minutes", label="pasta")
    with pytest.raises(NeedsClarificationError, match=r"Which one: 1\. eggs, 2\. pasta\?"):
        jarvis.prepare("timers.cancel", which="timer")
    with pytest.raises(ToolInputError, match="no timer or alarm matching “rice”") as info:
        jarvis.prepare("timers.cancel", which="rice")
    assert info.value.hint == "Pending: 1. eggs; 2. pasta"
    assert jarvis.run("timers.cancel", which="2").data["cancelled"] == [2]
    assert jarvis.run("timers.cancel", which="all").message == "Cancelled the “eggs” timer."


def test_cancelling_one_that_already_rang(jarvis: Harness) -> None:
    jarvis.run("timers.start", duration="1 minute", label="eggs")
    prepared = jarvis.prepare("timers.cancel", which="eggs")
    jarvis.store.mark_fired(1)  # the service rang it in the meantime
    assert prepared.run(threading.Event()).message == "It had already rung."
