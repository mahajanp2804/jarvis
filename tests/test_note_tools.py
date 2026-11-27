"""The notes.* tools on the fake Notes app."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.platforms.base import Note
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(
    runtime: Runtime, tool: str, context: ExecutionContext | None = None, **args: Any
) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args), context)


def notes(platform: FakePlatform) -> None:
    platform.notes_app.items = [
        Note("n1", "Shopping", "Notes", None, "Shopping\neggs\nmilk"),
        Note("n2", "Shopping for the trip", "Travel", None, "Shopping for the trip\nsocks"),
        Note("n3", "Ideas", "Notes", datetime(2026, 1, 5).astimezone(), "Ideas\nbuy a kayak"),
    ]


def test_search_and_read(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    notes(fake_platform)
    runtime, confirmer = make_runtime()
    found = run(runtime, "notes.search", words="kayak")
    assert (
        found.message
        == "Notes mentioning “kayak”:\n  1. Ideas (Notes, edited Mon 5 Jan): Ideas buy a kayak"
    )
    assert found.untrusted
    note = run(runtime, "notes.read", title="shopping")  # the exact title wins
    assert note.message == "“Shopping” (Notes):\nShopping\neggs\nmilk" and note.untrusted
    assert confirmer.prompts == []


def test_read_asks_when_unclear(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    notes(fake_platform)
    fake_platform.notes_app.items[0] = Note("n1", "Shopping list", "Notes", None, "x")
    runtime, _ = make_runtime()
    unclear = run(runtime, "notes.read", title="shopping")
    assert unclear.status is ToolStatus.NEEDS_CLARIFICATION
    assert "“Shopping list” (Notes), “Shopping for the trip” (Travel)" in unclear.message
    assert run(runtime, "notes.read", title="recipes").status is ToolStatus.FAILED


def test_create(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "notes.create", title="Packing", text="socks\ncharger")
    assert result.message == "Created the note “Packing” in Notes." and confirmer.prompts == []
    assert fake_platform.notes_app.items[0].text == "Packing\nsocks\ncharger"
    tainted = run(runtime, "notes.create", ExecutionContext(tainted=True), title="x")
    assert tainted.status is ToolStatus.CANCELLED  # after outside content, it asks
