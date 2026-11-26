from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(
    runtime: Runtime, tool: str, context: ExecutionContext | None = None, **args: Any
) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args), context)


def test_list_and_filter(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime()
    result = run(runtime, "shortcuts.list")
    assert result.ok and "Water Eject" in result.message and "Set Brightness" in result.message
    assert run(runtime, "shortcuts.list", query="bright").data["shortcuts"] == ["Set Brightness"]
    assert confirmer.prompts == []  # read-only


def test_running_asks_first_and_warns(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "shortcuts.run", name="set brightness", input="40")
    assert result.status is ToolStatus.CANCELLED and fake_platform.shortcut_runs == []
    assert "Run the shortcut “Set Brightness” with “40”" in confirmer.prompts[0]
    assert "can't see what a shortcut does" in confirmer.prompts[0]

    runtime, confirmer = make_runtime(confirm=True)
    assert run(runtime, "shortcuts.run", name="Set Brightness", input="40").ok
    assert fake_platform.shortcut_runs == [("Set Brightness", "40")]  # exact name used


def test_trusted_shortcuts_run_without_asking_until_outside_content(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False, shortcuts={"trusted": ["water eject"]})
    assert run(runtime, "shortcuts.run", name="Water Eject").ok
    assert confirmer.prompts == [] and fake_platform.shortcut_runs == [("Water Eject", None)]
    tainted = run(runtime, "shortcuts.run", ExecutionContext(tainted=True), name="Water Eject")
    assert tainted.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1


def test_output_is_outside_content(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    fake_platform.shortcut_output = "Extracted 3 files\n" + "x" * 3000
    runtime, _ = make_runtime(confirm=True)
    result = run(runtime, "shortcuts.run", name="Extract Audio")
    assert result.ok and result.untrusted
    assert "It returned:\nExtracted 3 files" in result.message
    assert result.message.endswith("[… more output not shown]")
    fake_platform.shortcut_output = ""
    assert not run(runtime, "shortcuts.run", name="Extract Audio").untrusted


def test_near_miss_asks_and_unknown_names_list_yours(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=True)
    near = run(runtime, "shortcuts.run", name="water ejec")
    assert near.status is ToolStatus.NEEDS_CLARIFICATION
    assert near.message == "There's no shortcut called “water ejec”. Did you mean “Water Eject”?"
    unknown = run(runtime, "shortcuts.run", name="Order Pizza")
    assert (
        unknown.status is ToolStatus.FAILED
        and "no shortcut called “Order Pizza”" in unknown.message
    )
    assert unknown.hint and "Water Eject" in unknown.hint
    assert fake_platform.shortcut_runs == [] and confirmer.prompts == []  # nothing ran or asked
