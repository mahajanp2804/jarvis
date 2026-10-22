"""Push-to-talk voice input: start a recording, stop it, get the transcript.

Nothing listens until you ask: each request starts a fresh worker process
(:mod:`jarvis.voice.stt_worker`) that opens the microphone, records until you
stop or pause, transcribes on-device with whisper.cpp, and exits. There is no
wake word and no background listening. The transcript is only text: callers
show it for confirmation (and editing) before anything runs, and it then goes
through the normal planner, policy and confirmations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from jarvis.errors import JarvisError
from jarvis.voice.models import DEFAULT_STT_MODEL, check_stt_model

log = logging.getLogger(__name__)

RESULT_TIMEOUT = 60.0  # the first transcription after an OS update compiles Metal shaders (~7 s)


class VoiceInputError(JarvisError):
    """Recording or transcription failed."""


@dataclass(frozen=True)
class Transcript:
    text: str
    audio_seconds: float
    transcribe_seconds: float
    reason: str  # why recording stopped: stop, silence, limit, no_speech, file


@dataclass(frozen=True)
class VoiceSettings:
    model_path: Path
    language: str = "en"
    max_seconds: float = 15.0
    silence_seconds: float = 1.5
    threads: int | None = None
    use_gpu: bool = False


class Recording:
    """One running worker. Thread-safe."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc
        self.listening = threading.Event()  # the microphone is open
        self.recorded = threading.Event()  # recording has stopped (any reason)
        self.done = threading.Event()  # transcript or error is available
        self._info: dict[str, Any] = {}
        self._text: str | None = None
        self._seconds = 0.0
        self._error: VoiceInputError | None = None
        self._cancelled = False
        threading.Thread(target=self._read, name="jarvis-voice-reader", daemon=True).start()

    def _read(self) -> None:
        stream: IO[str] | None = self._proc.stdout
        try:
            for line in stream or ():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("event")
                if kind == "listening":
                    self.listening.set()
                elif kind == "recorded":
                    self._info = event
                    self.recorded.set()
                elif kind == "text":
                    self._text = str(event.get("text", ""))
                    self._seconds = float(event.get("seconds", 0.0))
                elif kind == "error":
                    self._error = VoiceInputError(
                        str(event.get("message", "Voice input failed.")), hint=event.get("hint")
                    )
        finally:
            self._proc.wait()
            if self._text is None and self._error is None and not self._cancelled:
                self._error = VoiceInputError(
                    f"The voice worker stopped unexpectedly (exit {self._proc.returncode})."
                )
            self.listening.set()
            self.recorded.set()
            self.done.set()

    def stop(self) -> None:
        """Stop recording now and transcribe what was said."""
        stdin = self._proc.stdin
        if stdin is None:
            return
        try:
            stdin.write("stop\n")
            stdin.flush()
            stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass  # already finished

    def cancel(self) -> None:
        """Throw the recording away."""
        self._cancelled = True
        if self._proc.poll() is None:
            self._proc.kill()

    def result(self, timeout: float = RESULT_TIMEOUT) -> Transcript:
        if not self.done.wait(timeout):
            self.cancel()
            raise VoiceInputError(f"No transcript within {timeout:g}s.")
        if self._cancelled:
            raise VoiceInputError("Voice input was cancelled.")
        if self._error is not None:
            raise self._error
        return Transcript(
            text=self._text or "",
            audio_seconds=float(self._info.get("seconds", 0.0)),
            transcribe_seconds=self._seconds,
            reason=str(self._info.get("reason", "")),
        )


class VoiceInput:
    """Starts recordings with the configured model and limits."""

    def __init__(
        self,
        settings: VoiceSettings,
        *,
        command: Sequence[str] | None = None,
        log_file: Path | None = None,
    ) -> None:
        self.settings = settings
        self._command = (
            list(command) if command else [sys.executable, "-m", "jarvis.voice.stt_worker"]
        )
        self._log_file = log_file

    def check(self) -> None:
        """Fail fast on a missing or broken model file (before opening the microphone)."""
        path = self.settings.model_path
        check_stt_model(
            path, DEFAULT_STT_MODEL if path.name == DEFAULT_STT_MODEL.filename else None
        )

    def start(self, *, audio_file: Path | None = None) -> Recording:
        self.check()
        payload = {
            "model_path": str(self.settings.model_path),
            "language": self.settings.language,
            "max_seconds": self.settings.max_seconds,
            "silence_seconds": self.settings.silence_seconds,
            "threads": self.settings.threads,
            "use_gpu": self.settings.use_gpu,
            "audio_file": str(audio_file) if audio_file else None,
        }
        stderr: Any = subprocess.DEVNULL
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            stderr = self._log_file.open("w", encoding="utf-8")  # last run only, for debugging
        try:
            proc = subprocess.Popen(
                [*self._command, json.dumps(payload)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise VoiceInputError(f"Couldn't start voice input: {exc.strerror}") from exc
        finally:
            if stderr is not subprocess.DEVNULL:
                stderr.close()
        log.info("Voice worker %d started", proc.pid)
        return Recording(proc)
