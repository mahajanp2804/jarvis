"""Spoken replies on macOS with the built-in ``/usr/bin/say`` (on-device voices).

``say`` runs as a short-lived child process only while speaking, so silence
costs nothing. Text goes in through stdin, never as arguments, so it can't be
read as an option. Voices are the ones installed in System Settings >
Accessibility > Spoken Content.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path

from jarvis.errors import JarvisError
from jarvis.voice.speech import SpeechOutput, Voice

SAY = Path("/usr/bin/say")
# "Albert              en_US    # Hello!" and "Eddy (English (US)) en_US    # Hello!"
_VOICE_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<lang>[a-z]{2,3}_[A-Za-z0-9]+)\s+#")


def _env() -> dict[str, str]:
    keep = ("HOME", "LANG", "LC_ALL", "TMPDIR", "USER")
    return {"PATH": "/usr/bin:/bin", **{k: os.environ[k] for k in keep if k in os.environ}}


def list_voices(say: Path = SAY) -> list[Voice]:
    """Installed voices, from ``say -v '?'``."""
    try:
        out = subprocess.run(
            [str(say), "-v", "?"], capture_output=True, text=True, timeout=10, env=_env()
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise JarvisError(f"Can't list voices: {exc}") from exc
    voices = []
    for line in out.splitlines():
        if match := _VOICE_LINE.match(line):
            voices.append(Voice(match.group("name").strip(), match.group("lang")))
    return voices


class SaySpeaker(SpeechOutput):
    def __init__(
        self, voice: str | None = None, rate: int | None = None, *, say: Path = SAY
    ) -> None:
        self._say = say
        self._voice = voice or None
        self._rate = rate or None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None

    def check(self) -> None:
        """Raise a helpful error if ``say`` or the configured voice is missing."""
        if not self._say.is_file():
            raise JarvisError(f"{self._say} is missing, so Jarvis can't speak.")
        if self._voice:
            names = {v.name.casefold() for v in list_voices(self._say)}
            if self._voice.casefold() not in names:
                raise JarvisError(
                    f"The voice '{self._voice}' isn't installed.",
                    hint="See `jarvis voice voices`, or add voices in System Settings > "
                    "Accessibility > Spoken Content.",
                )

    def speak(self, text: str) -> None:
        text = text.strip()
        with self._lock:
            self._stop_locked()
            if not text:
                return
            command = [str(self._say)]
            if self._voice:
                command += ["-v", self._voice]
            if self._rate:
                command += ["-r", str(self._rate)]
            command += ["-f", "-"]  # read the text from stdin
            try:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_env(),
                    start_new_session=True,  # Ctrl+C in a terminal stops Jarvis, not mid-word
                )
            except OSError as exc:
                raise JarvisError(f"Can't start {self._say}: {exc.strerror}") from exc
            if proc.stdin is not None:
                try:
                    proc.stdin.write(text.encode("utf-8"))
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            self._proc = proc

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None:
            return True
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def voices(self) -> list[Voice]:
        return list_voices(self._say)
