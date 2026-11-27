"""The reminders.* tools on the fake Reminders app."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.platforms.base import Reminder
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(
    runtime: Runtime, tool: str, context: ExecutionContext | None = None, **args: Any
) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args), context)


@pytest.fixture
def fixed_now(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Saturday 28 Nov 2026, 4 PM local time."""
    now = datetime(2026, 11, 28, 16, 0).astimezone()
    import jarvis.tools.reminders as module

    original = module.ReminderTools.__init__

    def init(self: Any, platform: Any, clock: Any = None) -> None:
        original(self, platform, clock=lambda: now)

    monkeypatch.setattr(module.ReminderTools, "__init__", init)
    return now


def test_add_with_a_time_and_a_whole_day(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, fixed_now: datetime
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "reminders.add", title="buy  milk", when="tomorrow at 6pm")
    assert result.message == "Added the reminder “buy milk” to Reminders for tomorrow at 6:00 PM."
    assert confirmer.prompts == []  # LOW, like Siri
    added = fake_platform.reminder_app.items[0]
    assert added.due == datetime(2026, 11, 29, 18, 0).astimezone() and not added.all_day
    run(runtime, "reminders.add", title="pay rent", when="Dec 3", list="groceries")
    rent = fake_platform.reminder_app.items[1]
    assert rent.all_day and rent.list_name == "Groceries"  # the list's real name


def test_add_refuses_what_it_cannot_read(make_runtime: MakeRuntime, fixed_now: datetime) -> None:
    runtime, _ = make_runtime()
    for when in ("someday", "Feb 30"):
        result = run(runtime, "reminders.add", title="x", when=when)
        assert result.status is ToolStatus.FAILED
    near = run(runtime, "reminders.add", title="x", list="grocries")
    assert near.status is ToolStatus.NEEDS_CLARIFICATION and "“Groceries”" in near.message


def test_list_soonest_first_as_outside_content(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, fixed_now: datetime
) -> None:
    fake_platform.reminder_app.items = [
        Reminder("a", "Call Sam", "Reminders"),
        Reminder("b", "Buy milk", "Groceries", datetime(2026, 11, 28, 18, 0).astimezone()),
        Reminder("c", "Pay rent", "Reminders", datetime(2026, 12, 3).astimezone(), True),
    ]
    runtime, _ = make_runtime()
    result = run(runtime, "reminders.list")
    assert result.message == (
        "Open reminders (3):\n"
        "  1. Buy milk (today at 6:00 PM, Groceries)\n"
        "  2. Pay rent (Thu 3 Dec, Reminders)\n"
        "  3. Call Sam (Reminders)"
    )
    assert result.untrusted
    assert run(runtime, "reminders.list", list="Groceries").message.startswith(
        "Open reminders in Groceries (1):"
    )


def test_complete_by_title(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    fake_platform.reminder_app.items = [
        Reminder("a", "Call Sam", "Reminders"),
        Reminder("b", "Call Sam's mum", "Reminders"),
        Reminder("c", "Buy milk", "Groceries"),
    ]
    runtime, _ = make_runtime()
    assert run(runtime, "reminders.complete", title="milk").message == "Marked “Buy milk” as done."
    ambiguous = run(runtime, "reminders.complete", title="call")
    assert ambiguous.status is ToolStatus.NEEDS_CLARIFICATION
    assert "“Call Sam” (Reminders), “Call Sam's mum” (Reminders)" in ambiguous.message
    assert run(runtime, "reminders.complete", title="call sam").data["id"] == "a"  # exact wins
    assert run(runtime, "reminders.complete", title="dentist").status is ToolStatus.FAILED
    assert fake_platform.reminder_app.completed == ["c", "a"]


def test_changes_ask_after_outside_content(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "reminders.add", ExecutionContext(tainted=True), title="x")
    assert result.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1
