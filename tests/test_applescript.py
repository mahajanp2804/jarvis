"""The shared AppleScript runner (no app is controlled)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from jarvis.platforms.applescript import CLEAN, OSASCRIPT, records

needs_osascript = pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which(OSASCRIPT), reason="needs macOS AppleScript"
)


def test_records() -> None:
    text = "Buy milk\t2026-11-28 18:00\tGroceries\n\nCall Sam\n"
    assert records(text, 3) == [["Buy milk", "2026-11-28 18:00", "Groceries"], ["Call Sam", "", ""]]


@needs_osascript
def test_clean_replaces_separators_inside_values() -> None:
    """Runs only the helper: no app is involved."""
    script = CLEAN + "\non run argv\n\treturn clean(item 1 of argv)\nend run"
    result = subprocess.run(
        [OSASCRIPT, "-e", script, "a\tb\nc"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "a b c"
