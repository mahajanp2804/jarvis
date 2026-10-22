"""CLI handlers for ``jarvis voice ...`` (speech output; voice input from Phase 5 step 5)."""

from __future__ import annotations

import sys

from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.platforms.base import PlatformAdapter
from jarvis.voice.speech import SpeechOutput

EXIT_OK, EXIT_ERROR, EXIT_NOT_DONE = 0, 1, 3


def _print_error(exc: JarvisError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"  Hint: {exc.hint}", file=sys.stderr)


def speaker_for(config: JarvisConfig, platform: PlatformAdapter) -> SpeechOutput:
    voice = config.voice
    return platform.speech_output(voice.voice or None, voice.rate or None)


def voices(config: JarvisConfig, platform: PlatformAdapter, language: str | None) -> int:
    try:
        found = speaker_for(config, platform).voices()
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    if language:
        found = [v for v in found if v.language.lower().startswith(language.lower())]
    for voice in found:
        print(f"{voice.name:<28} {voice.language}")
    print(f'({len(found)} voice(s); set one with [voice] voice = "NAME")', file=sys.stderr)
    return EXIT_OK


def say(config: JarvisConfig, platform: PlatformAdapter, text: str) -> int:
    """Speak ``text`` with the configured voice and wait until it's done."""
    try:
        speaker = speaker_for(config, platform)
        speaker.check()
        speaker.speak(text)
        try:
            speaker.wait()
        except KeyboardInterrupt:
            speaker.stop()
            return 130
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    return EXIT_OK
