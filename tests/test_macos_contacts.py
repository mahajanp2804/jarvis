"""The Contacts adapter with a fake osascript (Contacts is never read)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jarvis.platforms.base import Contact
from jarvis.platforms.macos_contacts import ALL_SCRIPTS, FIND, MacContacts, tidy_label


def test_labels() -> None:
    assert tidy_label("_$!<Mobile>!$_") == "mobile"
    assert tidy_label("iPhone") == "iphone"
    assert tidy_label(" work ") == "work"


def test_find_parses_people_and_passes_words_as_arguments() -> None:
    calls: list[list[str]] = []

    def osascript(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        calls.append(argv)
        reply = (
            "p1\tJane Doe\t_$!<Mobile>!$_=+1 555 0100;home=+1 555 0199;\tjane@example.com;\n"
            "p1\tJane Doe\t_$!<Mobile>!$_=+1 555 0100;\t\n"  # found twice: name and relation
            "p2\tSam Lee\t\twork=sam@example.org;\n"
        )
        return subprocess.CompletedProcess(argv, 0, reply, "")

    people = MacContacts(osascript).find("mom", "mother")
    assert calls[0][2:] == [FIND, "mom", "mother"]
    assert people == [
        Contact("p1", "Jane Doe", (("mobile", "+1 555 0100"), ("home", "+1 555 0199")),
                (("", "jane@example.com"),)),
        Contact("p2", "Sam Lee", (), (("work", "sam@example.org"),)),
    ]  # fmt: skip
    MacContacts(osascript).find("Sam")
    assert calls[1][3:] == ["Sam", ""]


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Contacts no Apple events: it reads nothing."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert compiled.returncode == 0, compiled.stderr
