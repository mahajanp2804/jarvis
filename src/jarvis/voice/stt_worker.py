"""Voice-input worker: records one request from the microphone and transcribes it.

Runs as a short-lived child process, one per request::

    python -m jarvis.voice.stt_worker '<settings JSON>'

so the microphone, numpy, PortAudio and whisper.cpp never live in the
long-running service, and all of their memory goes back to the OS when the
request is done (measured: freeing whisper.cpp in-process kept ~85 MB). The
model loads while you speak, so it adds almost no wait.

Recording stops when the parent writes ``stop`` (or closes stdin), after
``silence_seconds`` of quiet following speech, or at ``max_seconds``. Audio is
kept in memory only and never written to disk. With ``audio_file`` (a 16 kHz
mono WAV) the worker transcribes that file instead of using the microphone,
for tests and `jarvis voice transcribe`.

Events go to stdout, one JSON object per line::

    {"event": "listening"}
    {"event": "recorded", "seconds": 2.4, "peak": 0.31, "speech": true, "reason": "silence"}
    {"event": "text", "text": "open Safari", "seconds": 0.08}
    {"event": "error", "kind": "missing|model|no_device|silent|failed", "message": "..."}
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from typing import Any

SAMPLE_RATE = 16_000
BLOCK_SECONDS = 0.1
NO_SPEECH_SECONDS = 8.0  # give up if nothing is said for this long
MIN_THRESHOLD = 0.01  # RMS level that counts as speech, at least


def emit(event: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
    sys.stdout.flush()


def _watch_stdin(stop: threading.Event) -> None:
    for line in sys.stdin:
        if line.strip() == "stop":
            break
    stop.set()  # "stop", or the parent closed stdin / went away


def _load_model(settings: dict[str, Any], box: dict[str, Any]) -> None:
    try:
        from pywhispercpp.model import Model

        box["model"] = Model(
            settings["model_path"],
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            print_realtime=False,
            n_threads=int(settings.get("threads") or min(4, os.cpu_count() or 4)),
            language=settings.get("language") or "en",
            no_context=True,
            single_segment=True,
        )
    except Exception as exc:  # reported to the parent, which shows it
        box["error"] = f"{type(exc).__name__}: {exc}"


def read_wav(path: str) -> tuple[Any, dict[str, Any]]:
    """A 16 kHz mono 16-bit WAV file as float32 samples, described like a recording."""
    import wave

    import numpy as np

    with wave.open(path, "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("expected a 16 kHz, mono, 16-bit WAV file")
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio, {
        "seconds": round(len(audio) / SAMPLE_RATE, 2),
        "peak": round(peak, 4),
        "speech": peak > MIN_THRESHOLD,
        "reason": "file",
    }


def resample(audio: Any, rate: int) -> Any:
    """Convert to 16 kHz: averaging for whole-number ratios (48 kHz), else interpolation."""
    import numpy as np

    if rate == SAMPLE_RATE or not len(audio):
        return audio.astype(np.float32)
    if rate % SAMPLE_RATE == 0:
        factor = rate // SAMPLE_RATE
        usable = len(audio) - len(audio) % factor
        return audio[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)
    count = int(len(audio) * SAMPLE_RATE / rate)
    positions = np.linspace(0, len(audio) - 1, count)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def record(settings: dict[str, Any], stop: threading.Event) -> tuple[Any, dict[str, Any]]:
    """Record until stopped, quiet after speech, or the time limit."""
    import numpy as np
    import sounddevice as sd

    # Record at the microphone's own rate (48 kHz on Macs) and convert afterwards.
    rate = int(sd.query_devices(kind="input")["default_samplerate"]) or SAMPLE_RATE

    blocks: queue.Queue[Any] = queue.Queue()

    def callback(data: Any, frames: int, when: Any, status: Any) -> None:
        blocks.put(data[:, 0].copy())

    max_seconds = float(settings.get("max_seconds", 15))
    silence_seconds = float(settings.get("silence_seconds", 1.5))
    chunks: list[Any] = []
    levels: list[float] = []
    speech = False
    quiet = 0.0
    reason = "limit"
    with sd.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        blocksize=int(rate * BLOCK_SECONDS),
        callback=callback,
    ):
        emit("listening")
        started = time.monotonic()
        while time.monotonic() - started < max_seconds:
            if stop.is_set():
                reason = "stop"
                break
            try:
                block = blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            chunks.append(block)
            level = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
            levels.append(level)
            # The first ~0.3 s is room noise; speech is clearly louder than that.
            noise = float(np.median(levels[:3]))
            threshold = max(MIN_THRESHOLD, noise * 3)
            if len(levels) > 3 and level > threshold:
                speech, quiet = True, 0.0
            elif speech:
                quiet += BLOCK_SECONDS
                if silence_seconds and quiet >= silence_seconds:
                    reason = "silence"
                    break
            elif time.monotonic() - started > NO_SPEECH_SECONDS:
                reason = "no_speech"
                break
    audio = resample(np.concatenate(chunks), rate) if chunks else np.zeros(0, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio, {
        "seconds": round(len(audio) / SAMPLE_RATE, 2),
        "peak": round(peak, 4),
        "speech": speech,
        "reason": reason,
    }


def main(argv: list[str]) -> int:
    settings: dict[str, Any] = json.loads(argv[1])
    try:
        import numpy  # noqa: F401
        import pywhispercpp  # noqa: F401
        import sounddevice
    except (ImportError, OSError) as exc:
        emit(
            "error",
            kind="missing",
            message=f"Voice input needs extra packages ({exc}).",
            hint='In the Jarvis virtualenv run: pip install -e ".[voice]"',
        )
        return 2

    stop = threading.Event()
    threading.Thread(target=_watch_stdin, args=(stop,), daemon=True).start()
    box: dict[str, Any] = {}
    loader = threading.Thread(target=_load_model, args=(settings, box), daemon=True)
    loader.start()  # load while the user speaks
    try:
        if settings.get("audio_file"):
            audio, info = read_wav(settings["audio_file"])
        else:
            audio, info = record(settings, stop)
    except sounddevice.PortAudioError as exc:
        emit("error", kind="no_device", message=f"No microphone is available ({exc}).")
        return 3
    except (OSError, EOFError, ValueError) as exc:  # unreadable audio file
        emit("error", kind="failed", message=f"Can't read the audio: {exc}")
        return 3
    emit("recorded", **info)
    if info["peak"] == 0.0:
        emit(
            "error",
            kind="silent",
            message="The microphone gave only silence.",
            hint="macOS may not allow Jarvis to use the microphone (System Settings > "
            "Privacy & Security > Microphone), or the input is muted.",
        )
        return 4
    if not info["speech"]:
        emit("text", text="", seconds=0.0)  # nothing was said; don't let the model guess
        return 0

    loader.join()
    if "error" in box:
        emit("error", kind="model", message=f"The speech model didn't load: {box['error']}")
        return 5
    started = time.monotonic()
    try:
        segments = box["model"].transcribe(audio)
    except Exception as exc:
        emit("error", kind="failed", message=f"Transcription failed: {exc}")
        return 6
    text = " ".join(s.text.strip() for s in segments).strip()
    emit("text", text=text, seconds=round(time.monotonic() - started, 3))
    return 0


if __name__ == "__main__":
    # Exit without running native destructors (whisper.cpp/Metal teardown is slow
    # and can crash at interpreter exit); the OS reclaims everything.
    code = main(sys.argv)
    sys.stdout.flush()
    os._exit(code)
