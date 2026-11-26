"""The music.* tools on the fake player (Music is never driven)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.platforms.base import NowPlaying
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(
    runtime: Runtime, tool: str, context: ExecutionContext | None = None, **args: Any
) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args), context)


def test_status(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime()
    assert run(runtime, "music.status").message == "Music isn't open."
    fake_platform.player.current = NowPlaying("paused", "So What", "Miles Davis", "Kind of Blue")
    result = run(runtime, "music.status")
    assert result.message == "Paused: “So What” by Miles Davis."
    assert result.untrusted and result.data["artist"] == "Miles Davis"  # names from the library
    assert confirmer.prompts == []


def test_controls_run_without_asking(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    assert (
        run(runtime, "music.control", command="play").message == "Playing “So What” by Miles Davis."
    )
    assert run(runtime, "music.control", command="next").message.startswith("Playing “Take Five”")
    assert run(runtime, "music.control", command="pause").message == "Paused."
    assert confirmer.prompts == [] and fake_platform.player.commands == ["play", "next", "pause"]


def test_nothing_to_do_never_starts_music(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime()
    for command in ("pause", "next", "previous"):
        result = run(runtime, "music.control", command=command)
        assert result.ok and result.message == "Nothing to do: Music isn't open."
    assert fake_platform.player.commands == []


def test_play_a_playlist(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "music.play", playlist="road trip playlist")
    assert (
        result.message
        == "Playing your “Road Trip” playlist. Playing “Take Five” by The Dave Brubeck Quartet."
    )
    assert fake_platform.player.commands == ["playlist:Road Trip"]  # the exact name


def test_playlist_near_misses_ask_and_unknown_ones_list_yours(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime()
    near = run(runtime, "music.play", playlist="road trp")
    assert near.status is ToolStatus.NEEDS_CLARIFICATION
    assert near.message == "There's no playlist called “road trp”. Did you mean “Road Trip”?"
    unknown = run(runtime, "music.play", playlist="Gym")
    assert unknown.status is ToolStatus.FAILED and unknown.hint and "Road Trip" in unknown.hint
    assert fake_platform.player.commands == []


def test_play_a_search(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, _ = make_runtime()
    assert (
        run(runtime, "music.play", search="miles davis").message
        == "Playing “So What” by Miles Davis."
    )
    missing = run(runtime, "music.play", search="polka")
    assert missing.status is ToolStatus.FAILED and "matches “polka”" in missing.message
    assert missing.hint and "not the Apple Music catalogue" in missing.hint


@pytest.mark.parametrize("args", [{}, {"playlist": "Focus", "search": "jazz"}, {"search": ""}])
def test_play_needs_one_argument(make_runtime: MakeRuntime, args: dict[str, Any]) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "music.play", **args)
    assert result.status is ToolStatus.FAILED and "nvalid" in result.message


def test_changes_ask_after_outside_content(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "music.play", ExecutionContext(tainted=True), search="jazz")
    assert result.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1
