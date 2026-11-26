"""Music.app control on macOS through AppleScript (``osascript``).

* **Automation permission**: the first command shows macOS's "Jarvis wants
  to control Music" prompt. Until it's allowed, commands fail with a
  permission error that says where to turn it on.
* **No text from a request is put into a script.** Playlist names and search
  words are passed as ``argv`` to fixed scripts, so they can't change what a
  script does.
* Reading what's playing, pausing and skipping never start Music; playing does.
* Only the user's own library: AppleScript can't search the Apple Music
  catalogue, so "play some jazz" plays the first library track that matches.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from jarvis.errors import OSPermissionError, ToolExecutionError
from jarvis.platforms.base import MusicPlayer, NowPlaying

OSASCRIPT = "/usr/bin/osascript"
# Long enough for the user to answer macOS's first Automation prompt.
TIMEOUT_SECONDS = 30

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

NOW_PLAYING = """\
if application "Music" is not running then return "not running"
tell application "Music"
	set s to player state as text
	if s is "stopped" then return s
	try
		set t to current track
		return s & linefeed & (name of t) & linefeed & (artist of t) & linefeed & (album of t)
	on error -- a radio stream has no current track
		return s
	end try
end tell"""

COMMANDS = {
    "play": 'tell application "Music" to play',
    "pause": 'if application "Music" is running then tell application "Music" to pause',
    "next": 'if application "Music" is running then tell application "Music" to next track',
    "previous": (
        'if application "Music" is running then tell application "Music" to previous track'
    ),
}

PLAYLISTS = """\
tell application "Music"
	set out to ""
	repeat with n in (get name of every user playlist)
		set out to out & n & linefeed
	end repeat
	return out
end tell"""

PLAY_PLAYLIST = """\
on run argv
	tell application "Music" to play user playlist (item 1 of argv)
end run"""

PLAY_SEARCH = """\
on run argv
	tell application "Music"
		set found to search library playlist 1 for (item 1 of argv)
		if found is {} then return ""
		set t to item 1 of found
		play t
		return (name of t) & linefeed & (artist of t) & linefeed & (album of t)
	end tell
end run"""

ALL_SCRIPTS = (NOW_PLAYING, *COMMANDS.values(), PLAYLISTS, PLAY_PLAYLIST, PLAY_SEARCH)

_NOT_ALLOWED = ("-1743", "not authorized to send apple events", "not allowed to send")


def parse_now_playing(text: str) -> NowPlaying:
    lines = text.rstrip("\n").split("\n")
    state = lines[0].strip() or "stopped"
    fields = [*(line.strip() for line in lines[1:4]), "", "", ""]
    return NowPlaying(state, fields[0], fields[1], fields[2])


class MacMusicPlayer(MusicPlayer):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._run = runner

    def _script(self, script: str, what: str, *args: str) -> str:
        try:
            proc = self._run(
                [OSASCRIPT, "-e", script, *args],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                f"Music didn't answer while trying to {what}.",
                hint="If macOS asked whether Jarvis may control Music, click OK and try again.",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"Couldn't {what}: {exc}") from exc
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or "").strip()
            if any(marker in error.lower() for marker in _NOT_ALLOWED):
                raise OSPermissionError(
                    "macOS doesn't allow Jarvis to control Music.",
                    hint="System Settings > Privacy & Security > Automation: turn on Music "
                    "for the app that runs Jarvis (Terminal, or Python for the service).",
                )
            raise ToolExecutionError(f"Couldn't {what}: {error or proc.returncode}")
        return proc.stdout or ""

    def now_playing(self) -> NowPlaying:
        return parse_now_playing(self._script(NOW_PLAYING, "read what's playing"))

    def command(self, command: str) -> None:
        if command not in COMMANDS:
            raise ValueError(f"unknown music command {command!r}")
        self._script(COMMANDS[command], f"{command} the music")

    def playlists(self) -> list[str]:
        listing = self._script(PLAYLISTS, "list your playlists")
        return [name.strip() for name in listing.splitlines() if name.strip()]

    def play_playlist(self, name: str) -> None:
        self._script(PLAY_PLAYLIST, f"play the playlist “{name}”", name)

    def play_search(self, query: str) -> NowPlaying | None:
        found = self._script(PLAY_SEARCH, f"play “{query}”", query)
        if not found.strip():
            return None
        return parse_now_playing("playing\n" + found)
