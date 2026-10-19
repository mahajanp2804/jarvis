from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.config import DEFAULT_TOOLS
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]
SCREEN_ON = {"enabled_tools": [*DEFAULT_TOOLS, "screen.*"]}


def run(runtime: Runtime, tool: str, **arguments: Any) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=arguments))


def test_screen_tools_are_off_by_default(make_runtime: MakeRuntime, home: Path) -> None:
    runtime, confirmer = make_runtime(confirm=True)
    result = run(runtime, "screen.read_text")
    assert result.status is ToolStatus.DENIED
    assert confirmer.prompts == []


def test_reading_the_screen_asks_and_deletes_the_capture(
    make_runtime: MakeRuntime, home: Path, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=True, policy=SCREEN_ON)
    result = run(runtime, "screen.read_text")
    assert result.status is ToolStatus.SUCCESS
    assert "Build passed\n3 warnings" in result.message
    assert result.untrusted is True
    assert "Capture the screen and read its text" in confirmer.prompts[0]
    assert "deleted right after reading" in confirmer.prompts[0]
    capture = fake_platform.captures[0]
    assert not capture.exists() and not capture.parent.exists()


def test_declined_capture_never_touches_the_screen(
    make_runtime: MakeRuntime, home: Path, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime(confirm=False, policy=SCREEN_ON)
    assert run(runtime, "screen.read_text").status is ToolStatus.CANCELLED
    assert fake_platform.captures == []


def test_reading_an_image_file_needs_no_capture_or_prompt(
    make_runtime: MakeRuntime, home: Path, fake_platform: FakePlatform
) -> None:
    (home / "Downloads" / "receipt.png").write_bytes(b"\x89PNG")
    runtime, confirmer = make_runtime(confirm=False, policy=SCREEN_ON)
    result = run(runtime, "screen.read_text", path="~/Downloads/receipt.png")
    assert result.status is ToolStatus.SUCCESS and "Build passed" in result.message
    assert confirmer.prompts == [] and fake_platform.captures == []


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("~/Library/x.png", ToolStatus.DENIED),
        ("/etc/x.png", ToolStatus.DENIED),
        ("~/Downloads/notes.txt", ToolStatus.FAILED),
        ("~/Downloads/missing.png", ToolStatus.FAILED),
    ],
)
def test_image_path_checks(
    make_runtime: MakeRuntime, home: Path, path: str, status: ToolStatus
) -> None:
    (home / "Downloads" / "notes.txt").write_text("x")
    runtime, _ = make_runtime(policy=SCREEN_ON)
    assert run(runtime, "screen.read_text", path=path).status is status


def test_missing_permission_explains_how_to_grant_it(
    make_runtime: MakeRuntime, home: Path, fake_platform: FakePlatform
) -> None:
    fake_platform.screen_allowed = False
    runtime, _ = make_runtime(confirm=True, policy=SCREEN_ON)
    result = run(runtime, "screen.read_text")
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "OSPermissionError" and result.hint == "grant it"


def test_capture_saves_to_desktop_and_never_overwrites(
    make_runtime: MakeRuntime, home: Path
) -> None:
    runtime, confirmer = make_runtime(confirm=True, policy=SCREEN_ON)
    saved = run(runtime, "screen.capture")
    assert saved.status is ToolStatus.SUCCESS and saved.verified
    assert saved.data["path"].startswith("~/Desktop/Jarvis screenshot ")
    explicit = run(runtime, "screen.capture", path="~/Documents/shot.png")
    assert explicit.status is ToolStatus.SUCCESS
    again = run(runtime, "screen.capture", path="~/Documents/shot.png")
    assert again.status is ToolStatus.FAILED and "never overwrites" in again.message
    outside = run(runtime, "screen.capture", path="~/Pictures/shot.png")
    assert outside.status is ToolStatus.DENIED
    assert len(confirmer.prompts) == 2
