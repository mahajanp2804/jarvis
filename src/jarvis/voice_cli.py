"""CLI handlers for ``jarvis voice ...``, ``jarvis listen`` and voice in ``jarvis shell``."""

from __future__ import annotations

import select
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from jarvis.ai.model_files import ModelFileError, download_model
from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.platforms.base import PlatformAdapter
from jarvis.voice.listen import Transcript, VoiceInput, VoiceSettings
from jarvis.voice.models import DEFAULT_STT_MODEL
from jarvis.voice.speech import SpeechOutput

EXIT_OK, EXIT_ERROR, EXIT_NOT_DONE = 0, 1, 3

InputFn = Callable[[str], str]
# Waits until the user presses Enter or ``finished`` is set; True if Enter was pressed.
EnterWaiter = Callable[[threading.Event], bool]


def _print_error(exc: JarvisError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"  Hint: {exc.hint}", file=sys.stderr)


# -- spoken replies ------------------------------------------------------------------


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


# -- voice input ---------------------------------------------------------------------


def voice_input(config: JarvisConfig) -> VoiceInput:
    voice = config.voice
    return VoiceInput(
        VoiceSettings(
            model_path=voice.resolved_stt_model_path(),
            language=voice.language,
            max_seconds=voice.max_record_seconds,
            silence_seconds=voice.silence_seconds,
        ),
        log_file=config.general.resolved_log_dir() / "voice-worker.log",
    )


def wait_for_enter(finished: threading.Event) -> bool:
    """Poll stdin (a terminal) so a pause can also end the recording."""
    while not finished.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()
            return True
    return False


def listen_once(voice: VoiceInput, waiter: EnterWaiter = wait_for_enter) -> Transcript:
    """Record one utterance in the terminal: stops on Enter or a pause."""
    recording = voice.start()
    try:
        if not recording.listening.wait(15):
            raise JarvisError("The microphone didn't start.")
        if not recording.done.is_set():
            print("Listening… speak now (it stops when you pause, or press Enter).", flush=True)
            if waiter(recording.recorded):
                recording.stop()
            print("Transcribing…", flush=True)
        return recording.result()
    except BaseException:
        recording.cancel()
        raise


def _prefilled_input(input_fn: InputFn, prompt: str, text: str) -> str:
    try:
        import readline
    except ImportError:  # pragma: no cover - always present on macOS
        return input_fn(prompt)
    readline.set_startup_hook(lambda: readline.insert_text(text))
    try:
        return input_fn(prompt)
    finally:
        readline.set_startup_hook()


def confirm_transcript(text: str, input_fn: InputFn | None = None) -> str | None:
    """Show what was heard; run it, edit it, or drop it. Returns the request or None."""
    ask = input_fn or input
    if not text:
        print("I didn't hear anything.")
        return None
    print(f'Heard: "{text}"')
    try:
        answer = ask("Run it? [Y]es / [e]dit / [n]o: ").strip().lower()
        if answer in {"", "y", "yes"}:
            return text
        if answer in {"e", "edit"}:
            edited = _prefilled_input(ask, "Request: ", text).strip()
            return edited or None
    except EOFError:
        pass
    return None


def listen(config: JarvisConfig) -> int:
    """`jarvis voice listen`: print what was heard, run nothing."""
    try:
        transcript = listen_once(voice_input(config))
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    print(f'Heard: "{transcript.text}"' if transcript.text else "I didn't hear anything.")
    print(
        f"({transcript.audio_seconds:.1f}s of audio, transcribed in "
        f"{transcript.transcribe_seconds:.2f}s; stopped: {transcript.reason})",
        file=sys.stderr,
    )
    return EXIT_OK if transcript.text else EXIT_NOT_DONE


def transcribe(config: JarvisConfig, audio_file: Path) -> int:
    """`jarvis voice transcribe FILE`: transcribe a 16 kHz mono WAV file."""
    try:
        transcript = voice_input(config).start(audio_file=audio_file).result()
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    print(transcript.text)
    return EXIT_OK if transcript.text else EXIT_NOT_DONE


def download(
    config: JarvisConfig,
    *,
    assume_yes: bool,
    input_fn: InputFn | None = None,
    downloader: Callable[..., Path] = download_model,
) -> int:
    spec = DEFAULT_STT_MODEL
    dest = config.voice.resolved_stt_model_path()
    if dest.name != spec.filename:
        print(f"[voice] stt_model_path points to {dest}; download that file yourself.")
        return EXIT_ERROR
    if dest.is_file() and dest.stat().st_size == spec.size_bytes:
        print(f"{spec.display_name} is already at {dest}.")
        return EXIT_OK
    print(
        f"Model:    {spec.display_name}\n"
        f"Licence:  {spec.license}  ({spec.page_url})\n"
        f"Size:     {spec.size_bytes / 1e6:.0f} MB\n"
        f"From:     {spec.url}\n"
        f"To:       {dest}",
        flush=True,
    )
    if not assume_yes:
        try:
            answer = (input_fn or input)("Download now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Not downloaded.")
            return EXIT_NOT_DONE
    try:
        downloader(spec, dest)
    except ModelFileError as exc:
        _print_error(exc)
        return EXIT_ERROR
    print(f"Downloaded and verified (SHA-256): {dest}")
    return EXIT_OK
