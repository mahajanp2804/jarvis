"""The alert sound for timers, with a fake afplay (no sound plays)."""

from __future__ import annotations

import subprocess
import threading
from typing import Any

from jarvis.platforms.macos import AFPLAY, MacOSPlatform


def test_alert_sound_plays_in_the_background() -> None:
    calls: list[list[str]] = []
    done = threading.Event()

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if len(calls) == 3:
            done.set()
        return subprocess.CompletedProcess(argv, 0, "", "")

    MacOSPlatform(runner=runner).play_alert("No Such Sound", repeats=3)
    assert done.wait(5)
    assert calls == [[AFPLAY, "/System/Library/Sounds/Glass.aiff"]] * 3  # falls back to Glass
