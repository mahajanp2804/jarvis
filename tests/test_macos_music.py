"""Music.app control with a fake osascript (Music is never driven)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jarvis.errors import OSPermissionError, ToolExecutionError
from jarvis.platforms.base import NowPlaying
from jarvis.platforms.macos_music import (
    ALL_SCRIPTS,
    COMMANDS,
    NOW_PLAYING,
    PLAY_PLAYLIST,
    PLAY_SEARCH,
    PLAYLISTS,
    MacMusicPlayer,
    parse_now_playing,
)


class FakeOsascript:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.replies: dict[str, tuple[int, str, str]] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        code, out, err = self.replies.get(argv[2], (0, "", ""))
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def osascript() -> FakeOsascript:
    return FakeOsascript()


def test_parse_now_playing() -> None:
    assert parse_now_playing("playing\nSo What\nMiles Davis\nKind of Blue\n") == NowPlaying(
        "playing", "So What", "Miles Davis", "Kind of Blue"
    )
    assert parse_now_playing("not running\n") == NowPlaying("not running")
    assert parse_now_playing("paused\n") == NowPlaying("paused")  # a radio stream


def test_reads_and_commands(osascript: FakeOsascript) -> None:
    osascript.replies[NOW_PLAYING] = (0, "paused\nSo What\nMiles Davis\nKind of Blue\n", "")
    player = MacMusicPlayer(osascript)
    assert player.now_playing().track == "So What"
    for command in ("play", "pause", "next", "previous"):
        player.command(command)
    assert [c[2] for c in osascript.calls[1:]] == list(COMMANDS.values())
    with pytest.raises(ValueError):
        player.command("shuffle")


def test_names_and_searches_go_in_argv_not_the_script(osascript: FakeOsascript) -> None:
    osascript.replies[PLAYLISTS] = (0, "Library\nRoad Trip\nFocus, Deep\n\n", "")
    osascript.replies[PLAY_SEARCH] = (0, "So What\nMiles Davis\nKind of Blue\n", "")
    player = MacMusicPlayer(osascript)
    assert player.playlists() == ["Library", "Road Trip", "Focus, Deep"]
    tricky = 'x" & (do shell script "id") & "'
    player.play_playlist(tricky)
    assert osascript.calls[-1] == ["/usr/bin/osascript", "-e", PLAY_PLAYLIST, tricky]
    playing = player.play_search("jazz")
    assert osascript.calls[-1][3:] == ["jazz"]
    assert playing == NowPlaying("playing", "So What", "Miles Davis", "Kind of Blue")
    osascript.replies[PLAY_SEARCH] = (0, "\n", "")
    assert player.play_search("polka") is None


def test_errors(osascript: FakeOsascript) -> None:
    osascript.replies[NOW_PLAYING] = (
        1, "", "execution error: Not authorized to send Apple events to Music. (-1743)",
    )  # fmt: skip
    with pytest.raises(OSPermissionError, match="control Music") as info:
        MacMusicPlayer(osascript).now_playing()
    assert info.value.hint and "Automation" in info.value.hint
    osascript.replies[PLAY_PLAYLIST] = (1, "", 'Can\'t get user playlist "Gone". (-1728)')
    with pytest.raises(ToolExecutionError, match="Couldn't play the playlist “Gone”"):
        MacMusicPlayer(osascript).play_playlist("Gone")


def test_a_hung_permission_prompt_times_out_with_a_hint() -> None:
    def hangs(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with pytest.raises(ToolExecutionError, match="Music didn't answer") as info:
        MacMusicPlayer(hangs).command("play")
    assert info.value.hint and "click OK" in info.value.hint


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Music no Apple events: it neither starts nor controls it."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert compiled.returncode == 0, compiled.stderr
