"""The Notes adapter with a fake osascript (Notes is never read or changed)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jarvis.errors import ToolExecutionError
from jarvis.platforms.macos_notes import ALL_SCRIPTS, CREATE, READ, SEARCH, MacNotes, note_html


class FakeOsascript:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.replies: dict[str, str] = {}

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, self.replies.get(argv[2], ""), "")


def test_note_html_escapes_everything() -> None:
    assert note_html("Shopping <list>", "eggs & milk\n\n<b>bread</b>") == (
        "<h1>Shopping &lt;list&gt;</h1><div>eggs &amp; milk</div><div><br></div>"
        "<div>&lt;b&gt;bread&lt;/b&gt;</div>"
    )


def test_search_newest_first() -> None:
    osascript = FakeOsascript()
    osascript.replies[SEARCH] = (
        "n1\tWifi password ideas\tNotes\t2026-01-05T10:00\tnot really\n"
        "n2\tShopping\tNotes\t2026-11-27T09:00\teggs, milk\n"
    )
    found = MacNotes(osascript).search("shopping", limit=5)
    assert osascript.calls[0][3:] == ["shopping", "5"]
    assert [n.title for n in found] == ["Shopping", "Wifi password ideas"]
    assert found[0].text == "eggs, milk"


def test_read_and_locked_notes() -> None:
    osascript = FakeOsascript()
    osascript.replies[READ] = "n2\tShopping\tNotes\t2026-11-27T09:00\nShopping\neggs\nmilk\n"
    note = MacNotes(osascript).read("n2")
    assert (note.title, note.folder, note.text) == ("Shopping", "Notes", "Shopping\neggs\nmilk")
    osascript.replies[READ] = "#locked\n"
    with pytest.raises(ToolExecutionError, match="locked"):
        MacNotes(osascript).read("n3")


def test_create_passes_escaped_html_as_an_argument() -> None:
    osascript = FakeOsascript()
    osascript.replies[CREATE] = "n9\tPacking\tNotes\n"
    note = MacNotes(osascript).create("Packing", 'socks" & (do shell script "id") & "')
    assert osascript.calls[0][2] == CREATE
    assert osascript.calls[0][3] == (
        "<h1>Packing</h1><div>socks&quot; &amp; (do shell script &quot;id&quot;) &amp; &quot;</div>"
    )
    assert osascript.calls[0][4] == "" and (note.id, note.folder) == ("n9", "Notes")


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Notes no Apple events: it reads and changes nothing."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert compiled.returncode == 0, compiled.stderr
