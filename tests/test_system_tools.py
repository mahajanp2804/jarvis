"""The system.* tools on the fake platform (nothing on this Mac changes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(
    runtime: Runtime, tool: str, context: ExecutionContext | None = None, **args: Any
) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args), context)


def test_status(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime()
    result = run(runtime, "system.status")
    assert result.ok and confirmer.prompts == []
    assert result.message == "Volume 50% · Wi-Fi on · Battery 80%, on battery · Light mode"
    assert result.data["battery_percent"] == 80


def test_volume_changes_without_asking(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    assert run(runtime, "system.volume", level=30).message == "The volume is now 30% (was 50%)."
    assert run(runtime, "system.volume", change="up").message == "The volume is now 40% (was 30%)."
    assert run(runtime, "system.volume", mute=True).message == "The sound is now muted."
    assert run(runtime, "system.volume", mute=True).message == "The sound is already muted."
    assert confirmer.prompts == []  # LOW: like Siri, no confirmation
    assert fake_platform.system.changes == [("volume", 30), ("volume", 40), ("muted", True)]


def test_volume_limits_and_unmuting(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    system = fake_platform.system
    system.current = replace(system.current, volume=95, muted=True)
    runtime, _ = make_runtime()
    run(runtime, "system.volume", change="down")  # stays muted
    assert system.current.muted is True and system.current.volume == 85
    result = run(runtime, "system.volume", change="up")  # up also unmutes
    assert result.message == "The volume is now 95% (was 85%)." and system.current.muted is False
    assert run(runtime, "system.volume", change="up").data["volume"] == 100  # capped
    assert run(runtime, "system.volume", change="up").message == "The volume is already 100%."


@pytest.mark.parametrize(
    "args", [{}, {"level": 20, "mute": True}, {"level": 150}, {"change": "louder"}]
)
def test_volume_rejects_bad_arguments(make_runtime: MakeRuntime, args: dict[str, Any]) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "system.volume", **args)
    assert result.status is ToolStatus.FAILED and "nvalid" in result.message


def test_outputs_without_volume_control(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.system.current = replace(fake_platform.system.current, volume=None)
    runtime, _ = make_runtime()
    result = run(runtime, "system.volume", level=20)
    assert result.status is ToolStatus.FAILED and "can't be set from the Mac" in result.message
    assert fake_platform.system.changes == []


def test_wifi_off_asks_first_and_on_does_not(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    declined = run(runtime, "system.wifi", on=False)
    assert declined.status is ToolStatus.CANCELLED and fake_platform.system.changes == []
    assert "Turn Wi-Fi off" in confirmer.prompts[0] and "goes offline" in confirmer.prompts[0]

    runtime, confirmer = make_runtime(confirm=True)
    assert run(runtime, "system.wifi", on=False).message == "Turned Wi-Fi off."
    runtime, confirmer = make_runtime(confirm=False)
    assert run(runtime, "system.wifi", on=True).message == "Turned Wi-Fi on."
    assert run(runtime, "system.wifi", on=True).message == "Wi-Fi is already on."
    assert confirmer.prompts == []
    assert fake_platform.system.changes == [("wifi", False), ("wifi", True)]


def test_changes_ask_after_outside_content(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "system.volume", ExecutionContext(tainted=True), level=100)
    assert result.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1


def test_sleep_display(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    assert run(runtime, "system.sleep_display").message == "Turned the display off."
    assert confirmer.prompts == [] and fake_platform.system.changes == [("sleep_display", None)]
