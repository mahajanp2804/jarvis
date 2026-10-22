"""`jarvis voice benchmark`: speech-to-text speed and accuracy, without a microphone.

Each test command is synthesized with a system voice (``say``, converted to a
16 kHz mono WAV with ``afconvert``) and transcribed by the real voice worker: a
fresh process per command, exactly what push-to-talk runs once you stop
talking. Reported per command: whether the transcript matches (ignoring case
and punctuation), wall time for the whole worker, transcription time, and the
worker's CPU time; overall, the peak memory of the largest worker.

Synthetic speech is clean and evenly paced, so the accuracy here is an upper
bound for a real microphone. Try other voices (``--voice Rishi``,
``--voice Karen``) to see how accents fare.
"""

from __future__ import annotations

import json
import re
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jarvis.errors import JarvisError
from jarvis.voice.listen import VoiceInput

DEFAULT_SENTENCES = (
    "open Safari and the Notes app",
    "open my Stock Price Predictor project and run it",
    "what's on the to-do list on my desktop?",
    "remember that my editor is Visual Studio Code",
    "run the morning setup workflow",
    "quit Music",
)

SAY = Path("/usr/bin/say")
AFCONVERT = Path("/usr/bin/afconvert")

Synthesizer = Callable[[str, Path, "str | None"], None]


@dataclass(frozen=True)
class VoiceSample:
    sentence: str
    heard: str
    exact: bool
    wall_seconds: float
    transcribe_seconds: float
    cpu_seconds: float


@dataclass(frozen=True)
class VoiceReport:
    samples: list[VoiceSample]
    peak_worker_mb: float | None
    use_gpu: bool
    voice: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": [asdict(s) for s in self.samples],
            "peak_worker_mb": self.peak_worker_mb,
            "use_gpu": self.use_gpu,
            "voice": self.voice,
        }


def normalise(text: str) -> str:
    """'What's on the To-Do list?' -> "what's on the to do list"."""
    text = text.lower().replace("-", " ")
    return " ".join(re.sub(r"[^\w' ]+", " ", text).split())


def synthesize(sentence: str, dest: Path, voice: str | None) -> None:
    """Speak ``sentence`` into ``dest`` as a 16 kHz mono 16-bit WAV (macOS)."""
    if not (SAY.is_file() and AFCONVERT.is_file()):
        raise JarvisError("The voice benchmark synthesizes speech with macOS's say and afconvert.")
    aiff = dest.with_suffix(".aiff")
    command = [str(SAY), *(["-v", voice] if voice else []), "-o", str(aiff), "-f", "-"]
    convert = [str(AFCONVERT), "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(dest)]
    try:
        subprocess.run(command, input=sentence.encode(), check=True, timeout=60)
        subprocess.run(convert, check=True, timeout=60, capture_output=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise JarvisError(f"Couldn't synthesize test speech: {exc}") from exc


def _children_usage() -> tuple[float, int]:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime, usage.ru_maxrss


def run_voice_benchmark(
    voice_input: VoiceInput,
    workdir: Path,
    *,
    sentences: Sequence[str] = DEFAULT_SENTENCES,
    speaker_voice: str | None = None,
    synth: Synthesizer = synthesize,
    progress: Callable[[str], None] = lambda _msg: None,
) -> VoiceReport:
    samples = []
    peak = 0
    for index, sentence in enumerate(sentences, start=1):
        wav = workdir / f"{index}.wav"
        synth(sentence, wav, speaker_voice)
        progress(f"{index}/{len(sentences)}: {sentence}")
        cpu_before, _ = _children_usage()
        started = time.monotonic()
        transcript = voice_input.start(audio_file=wav).result()
        wall = time.monotonic() - started
        cpu_after, maxrss = _children_usage()  # the worker has exited and been reaped
        peak = max(peak, maxrss)
        samples.append(
            VoiceSample(
                sentence=sentence,
                heard=transcript.text,
                exact=normalise(transcript.text) == normalise(sentence),
                wall_seconds=round(wall, 3),
                transcribe_seconds=round(transcript.transcribe_seconds, 3),
                cpu_seconds=round(cpu_after - cpu_before, 3),
            )
        )
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 1024
    return VoiceReport(
        samples,
        round(peak_mb, 1) if peak else None,
        voice_input.settings.use_gpu,
        speaker_voice,
    )


def render(report: VoiceReport) -> str:
    samples = report.samples
    if not samples:
        return "No samples."
    lines = [
        f"Speech-to-text on {len(samples)} synthesized commands "
        f"({report.voice or 'system voice'}; whisper on the {'GPU' if report.use_gpu else 'CPU'}):",
        "",
        f"  {'match':<6}{'wall':>7}{'stt':>7}{'cpu':>7}  heard",
    ]
    for s in samples:
        mark = "yes" if s.exact else "NO"
        lines.append(
            f"  {mark:<6}{s.wall_seconds:>6.2f}s{s.transcribe_seconds:>6.2f}s"
            f"{s.cpu_seconds:>6.2f}s  {s.heard!r}"
        )
        if not s.exact:
            lines.append(f"  {'':<27}  said: {s.sentence!r}")
    exact = sum(s.exact for s in samples)
    walls = [s.wall_seconds for s in samples]
    stt = statistics.median(s.transcribe_seconds for s in samples)
    cpu = statistics.median(s.cpu_seconds for s in samples)
    lines += [
        "",
        f"Exact transcripts:      {exact}/{len(samples)}",
        f"Wall time per request:  median {statistics.median(walls):.2f}s "
        f"(first {walls[0]:.2f}s, max {max(walls):.2f}s); the worker process start included",
        f"Transcription alone:    median {stt:.2f}s",
        f"CPU per request:        median {cpu:.2f}s",
    ]
    if report.peak_worker_mb is not None:
        lines.append(
            f"Peak worker memory:     {report.peak_worker_mb:.0f} MB (freed when it exits)"
        )
    lines.append(
        "Synthetic speech is clean: treat accuracy as an upper bound for a real microphone."
    )
    return "\n".join(lines)


def write_json(report: VoiceReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
