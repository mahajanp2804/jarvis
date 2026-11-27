"""The calendar.* tools on the fake Calendar app."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.platforms.base import CalendarEvent
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def local(*parts: int) -> datetime:
    return datetime(*parts).astimezone()  # type: ignore[arg-type]


NOW = local(2026, 11, 28, 16, 0)  # Saturday


@pytest.fixture(autouse=True)
def fixed_now(monkeypatch: pytest.MonkeyPatch) -> None:
    import jarvis.tools.calendar as module

    original = module.CalendarTools.__init__

    def init(self: Any, platform: Any, clock: Any = None) -> None:
        original(self, platform, clock=lambda: NOW)

    monkeypatch.setattr(module.CalendarTools, "__init__", init)


def run(runtime: Runtime, tool: str, **args: Any) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args))


@pytest.fixture
def week(fake_platform: FakePlatform) -> FakePlatform:
    fake_platform.calendar_app.items = [
        CalendarEvent(
            "1", "Mom's birthday", "Birthdays", local(2026, 11, 29), local(2026, 11, 30), True
        ),
        CalendarEvent(
            "2", "Standup", "Work", local(2026, 11, 30, 9, 30), local(2026, 11, 30, 9, 45)
        ),
        CalendarEvent(
            "3",
            "Dentist",
            "Home",
            local(2026, 11, 30, 11, 30),
            local(2026, 11, 30, 13, 0),
            location="Main St",
        ),
    ]
    return fake_platform


def test_a_day(make_runtime: MakeRuntime, week: FakePlatform) -> None:
    runtime, confirmer = make_runtime()
    result = run(runtime, "calendar.events", when="monday")
    assert result.message == (
        "Your calendar on Mon 30 Nov (2):\n"
        "  9:30 to 9:45 AM: Standup (Work)\n"
        "  11:30 AM to 1:00 PM: Dentist, Main St (Home)"
    )
    assert result.untrusted and confirmer.prompts == []
    assert week.calendar_app.windows[-1] == (local(2026, 11, 30), local(2026, 12, 1))
    assert run(runtime, "calendar.events").message == "Nothing on your calendar today."


def test_a_week_is_grouped_by_day(make_runtime: MakeRuntime, week: FakePlatform) -> None:
    week.calendar_app.notes = ["Repeating events in “Work” couldn't be read and may be missing."]
    runtime, _ = make_runtime()
    message = run(runtime, "calendar.events", when="next week").message
    assert message.startswith("Your calendar next week (2):\nMon 30 Nov:\n  9:30 to 9:45 AM")
    assert message.endswith("Note: Repeating events in “Work” couldn't be read and may be missing.")
    assert (
        "Tomorrow:\n  All day: Mom's birthday (Birthdays)"
        in run(runtime, "calendar.events", when="this week").message
    )


def test_adding_asks_first_with_the_exact_time(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    declined = run(
        runtime, "calendar.add", title="Lunch with Sam", when="friday at 1", location="Cafe"
    )
    assert declined.status is ToolStatus.CANCELLED and fake_platform.calendar_app.items == []
    assert (
        "Add “Lunch with Sam” to your calendar: Fri 4 Dec, 1:00 to 2:00 PM at Cafe"
        in (confirmer.prompts[0])
    )
    runtime, _ = make_runtime(confirm=True)
    added = run(
        runtime,
        "calendar.add",
        title="Call",
        when="tomorrow 9:30am",
        duration="30 minutes",
        calendar="work",
    )
    assert added.message == "Added “Call” to Work: tomorrow, 9:30 to 10:00 AM."
    event = fake_platform.calendar_app.items[0]
    assert (event.start, event.end, event.calendar) == (
        local(2026, 11, 29, 9, 30),
        local(2026, 11, 29, 10, 0),
        "Work",
    )


def test_whole_day_events_and_bad_input(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime(confirm=True)
    assert run(runtime, "calendar.add", title="Holiday", when="Dec 24").message == (
        "Added “Holiday” to Home: Thu 24 Dec, all day."
    )
    assert fake_platform.calendar_app.items[0].all_day
    for when in ("someday", "Feb 30"):
        assert run(runtime, "calendar.add", title="x", when=when).status is ToolStatus.FAILED
    near = run(runtime, "calendar.add", title="x", when="monday at 9", calendar="hme")
    assert near.status is ToolStatus.NEEDS_CLARIFICATION and "“Home”" in near.message
