"""Messages and Mail adapters with a fake osascript (nothing is ever sent)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jarvis.errors import ToolExecutionError
from jarvis.platforms.macos_messages import (
    ALL_SCRIPTS,
    COMPOSE_MAIL,
    SEND_MESSAGE,
    MacMail,
    MacMessages,
)


class FakeOsascript:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.failing = failing or set()  # services that fail

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        if argv[-1] in self.failing:
            return subprocess.CompletedProcess(argv, 1, "", f"{argv[-1]}: Can't send. (-1728)")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_imessage_first() -> None:
    osascript = FakeOsascript()
    tricky = 'hi" & (do shell script "id") & "'
    assert MacMessages(osascript).send("+1 555 0100", tricky) == "iMessage"
    assert osascript.calls == [
        ["/usr/bin/osascript", "-e", SEND_MESSAGE, "+1 555 0100", tricky, "iMessage"]
    ]


def test_sms_when_imessage_fails() -> None:
    osascript = FakeOsascript(failing={"iMessage"})
    assert MacMessages(osascript).send("+1 555 0100", "hi") == "SMS"
    assert [c[-1] for c in osascript.calls] == ["iMessage", "SMS"]


def test_the_imessage_error_is_reported_when_both_fail() -> None:
    osascript = FakeOsascript(failing={"iMessage", "SMS"})
    with pytest.raises(ToolExecutionError, match="iMessage: Can't send"):
        MacMessages(osascript).send("+1 555 0100", "hi")


def test_mail_sends_or_opens_a_draft() -> None:
    osascript = FakeOsascript()
    MacMail(osascript).compose("sam@example.org", "Sick today", "I'm unwell.", send=False)
    MacMail(osascript).compose("sam@example.org", "Sick today", "I'm unwell.", send=True)
    assert osascript.calls[0][2:] == [
        COMPOSE_MAIL,
        "sam@example.org",
        "Sick today",
        "I'm unwell.",
        "0",
    ]
    assert osascript.calls[1][-1] == "1"


@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS AppleScript")
@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_scripts_compile(script: str) -> None:
    """Compiling sends Messages and Mail no Apple events: nothing is sent."""
    compiled = subprocess.run(
        ["/usr/bin/osacompile", "-o", "/dev/null", "-e", script],
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert compiled.returncode == 0, compiled.stderr
