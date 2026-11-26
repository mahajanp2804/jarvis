"""Music tools: what's playing, play/pause/skip, and play a playlist or song.

* All LOW risk, like Siri's music commands: easily undone, nothing is lost.
* ``music.play`` plays a playlist by name, or the first song in the user's
  library matching a search. AppleScript can't reach the Apple Music
  catalogue, so songs that aren't in the library can't be played.
* Playlist names must match one of the user's playlists; a near miss gets
  "Did you mean …?" instead of a guess.
* Song, artist and album names come from the library (and so from outside
  Jarvis): results that contain them are marked as outside content.
"""

from __future__ import annotations

import difflib
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.platforms.base import MusicPlayer, NowPlaying, PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec


class StatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Literal["play", "pause", "next", "previous"] = Field(
        description="play, pause, next or previous"
    )


class PlayArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    playlist: str | None = Field(default=None, min_length=1, max_length=200)
    search: str | None = Field(
        default=None, min_length=1, max_length=200, description="song, artist, album or genre"
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> PlayArgs:
        if (self.playlist is None) == (self.search is None):
            raise ValueError("give either playlist or search")
        return self


def describe(playing: NowPlaying) -> str:
    if playing.state == "not running":
        return "Music isn't open."
    if not playing.track:
        return "Nothing is playing." if playing.state == "stopped" else f"Music is {playing.state}."
    by = f" by {playing.artist}" if playing.artist else ""
    verb = "Playing" if playing.state == "playing" else playing.state.capitalize() + ":"
    return f"{verb} “{playing.track}”{by}."


def _result(tool: str, message: str, playing: NowPlaying) -> ToolResult:
    return ToolResult(
        tool=tool,
        status=ToolStatus.SUCCESS,
        message=message,
        data={"state": playing.state, "track": playing.track, "artist": playing.artist},
        untrusted=bool(playing.track),  # names come from the library
    )


class MusicTools:
    """Builds the ``music.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    def _player(self) -> MusicPlayer:
        return self._platform.music()  # UnsupportedPlatformError off macOS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "music.status",
                "Show what's playing in Music.",
                StatusArgs,
                RiskLevel.READ_ONLY,
                self._prepare_status,
                idempotent=True,
            ),
            ToolSpec(
                "music.control",
                "Play, pause, skip to the next song or go back in Music.",
                ControlArgs,
                RiskLevel.LOW,
                self._prepare_control,
            ),
            ToolSpec(
                "music.play",
                "Play one of the user's playlists, or search their Music library and play "
                "the first match. Give playlist or search.",
                PlayArgs,
                RiskLevel.LOW,
                self._prepare_play,
            ),
        ]

    def _prepare_status(self, args: StatusArgs) -> PreparedAction:
        player = self._player()

        def run(cancel: threading.Event) -> ToolResult:
            playing = player.now_playing()
            return _result("music.status", describe(playing), playing)

        return PreparedAction(description="Check what's playing", run=run)

    def _prepare_control(self, args: ControlArgs) -> PreparedAction:
        player = self._player()
        before = player.now_playing()  # never starts Music
        command = args.command
        already = {
            "play": before.state == "playing",
            "pause": before.state != "playing",
            "next": before.state == "not running",
            "previous": before.state == "not running",
        }[command]
        labels = {"play": "Play", "pause": "Pause", "next": "Skip to the next song",
                  "previous": "Go back a song"}  # fmt: skip

        def run(cancel: threading.Event) -> ToolResult:
            if already:
                return _result("music.control", "Nothing to do: " + describe(before), before)
            player.command(command)
            after = player.now_playing()
            message = "Paused." if command == "pause" else describe(after)
            return _result("music.control", message, after)

        return PreparedAction(description=f"{labels[command]} in Music", run=run, noop=already)

    def _prepare_play(self, args: PlayArgs) -> PreparedAction:
        player = self._player()
        playlist, search = args.playlist, args.search or ""  # exactly one is given

        def run(cancel: threading.Event) -> ToolResult:
            if playlist is not None:
                name = self._resolve_playlist(player, playlist)  # starts Music
                player.play_playlist(name)
                playing = player.now_playing()
                return _result(
                    "music.play", f"Playing your “{name}” playlist. {describe(playing)}", playing
                )
            found = player.play_search(search)
            if found is None:
                raise ToolExecutionError(
                    f"Nothing in your Music library matches “{search}”.",
                    hint="Jarvis can play songs in your library, not the Apple Music catalogue.",
                )
            return _result("music.play", describe(found), found)

        what = f"the playlist “{playlist}”" if playlist else f"music matching “{search}”"
        return PreparedAction(description=f"Play {what} in Music", run=run)

    @staticmethod
    def _resolve_playlist(player: MusicPlayer, wanted: str) -> str:
        names = player.playlists()
        by_key = {n.casefold(): n for n in names}
        key = wanted.casefold().removesuffix(" playlist").strip()
        if key in by_key:
            return by_key[key]
        close = difflib.get_close_matches(key, list(by_key), n=2, cutoff=0.75)
        if close:
            options = " or ".join(f"“{by_key[c]}”" for c in close)
            raise NeedsClarificationError(
                f"There's no playlist called “{wanted}”. Did you mean {options}?"
            )
        known = ", ".join(names[:15]) + (" …" if len(names) > 15 else "")
        raise ToolExecutionError(
            f"There's no playlist called “{wanted}”.",
            hint=f"Your playlists: {known}." if names else "Your Music library has no playlists.",
        )
