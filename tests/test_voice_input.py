from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis import main as cli
from jarvis import voice_cli
from jarvis.ai.model_files import ModelFileError
from jarvis.bootstrap import Runtime
from jarvis.config import JarvisConfig
from jarvis.voice.listen import Transcript, VoiceInput, VoiceInputError, VoiceSettings
from jarvis.voice.models import DEFAULT_STT_MODEL, GGML_MAGIC
from tests.fakes import FakePlatform, ScriptedConfirmer
from tests.test_assistant import ScriptedUI, drive, make

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]
FAKE_WORKER = Path(__file__).with_name("fake_voice_worker.py")


def fake_model(tmp_path: Path) -> Path:
    path = tmp_path / "ggml-test.bin"
    path.write_bytes(GGML_MAGIC + b"\0" * 64)
    return path


def fake_voice(tmp_path: Path, scenario: str = "ok", max_seconds: float = 5) -> VoiceInput:
    return VoiceInput(
        VoiceSettings(fake_model(tmp_path), max_seconds=max_seconds),
        command=[sys.executable, str(FAKE_WORKER), scenario],
    )


# -- recordings ---------------------------------------------------------------------------


def test_stop_ends_recording_and_returns_the_transcript(tmp_path: Path) -> None:
    recording = fake_voice(tmp_path).start()
    assert recording.listening.wait(5)
    assert not recording.recorded.is_set()  # still listening until told
    recording.stop()
    transcript = recording.result(timeout=5)
    assert transcript == Transcript("open safari", 2.0, 0.05, "stop")


def test_the_log_says_how_attempts_end_but_never_what_was_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice.listen")
    recording = fake_voice(tmp_path).start()
    assert recording.listening.wait(5)
    recording.stop()
    recording.result(timeout=5)
    for scenario in ("nothing", "silent"):
        with contextlib.suppress(VoiceInputError):
            fake_voice(tmp_path, scenario).start().result(timeout=5)
    cancelled = fake_voice(tmp_path).start()
    assert cancelled.listening.wait(5)
    cancelled.cancel()
    assert cancelled.done.wait(5)
    ended = [r.getMessage() for r in caplog.records if "ended:" in r.getMessage()]
    assert any("heard 11 characters (2.0 s of audio, stopped: stop)" in m for m in ended)
    assert any("nothing heard (8.0 s of audio, stopped: no_speech)" in m for m in ended)
    assert any("failed: The microphone gave only silence." in m for m in ended)
    assert any(m.endswith("ended: cancelled") for m in ended)
    assert "open safari" not in caplog.text


def test_cancel_discards_the_recording(tmp_path: Path) -> None:
    recording = fake_voice(tmp_path).start()
    assert recording.listening.wait(5)
    recording.cancel()
    with pytest.raises(VoiceInputError, match="cancelled"):
        recording.result(timeout=5)


@pytest.mark.parametrize(
    ("scenario", "message"),
    [("silent", "only silence"), ("crash", "stopped unexpectedly")],
)
def test_worker_failures_are_reported(tmp_path: Path, scenario: str, message: str) -> None:
    recording = fake_voice(tmp_path, scenario).start()
    with pytest.raises(VoiceInputError, match=message) as info:
        recording.result(timeout=5)
    if scenario == "silent":
        assert info.value.hint == "grant it"
    assert recording.recorded.is_set()  # a dialog waiting on it closes


def test_nothing_said_gives_empty_text(tmp_path: Path) -> None:
    assert fake_voice(tmp_path, "nothing").start().result(timeout=5).text == ""


def test_missing_or_broken_model_fails_before_recording(tmp_path: Path) -> None:
    voice = VoiceInput(VoiceSettings(tmp_path / DEFAULT_STT_MODEL.filename))
    with pytest.raises(ModelFileError, match="Speech model not found") as info:
        voice.start()
    assert info.value.hint and "jarvis voice download" in info.value.hint
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"GGUF....")
    with pytest.raises(ModelFileError, match=r"not a whisper\.cpp"):
        VoiceInput(VoiceSettings(bad)).start()
    short = tmp_path / DEFAULT_STT_MODEL.filename
    short.write_bytes(GGML_MAGIC + b"\0" * 10)
    with pytest.raises(ModelFileError, match="incomplete"):
        VoiceInput(VoiceSettings(short)).start()


def test_wav_reader_checks_the_format(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    import wave

    from jarvis.voice.stt_worker import read_wav

    good = tmp_path / "good.wav"
    with wave.open(str(good), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((np.sin(np.arange(16_000) / 5) * 8000).astype(np.int16).tobytes())
    audio, info = read_wav(str(good))
    assert len(audio) == 16_000 and info["speech"] and info["seconds"] == 1.0
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44_100)
        wav.writeframes(b"\0\0\0\0" * 10)
    with pytest.raises(ValueError, match="16 kHz, mono"):
        read_wav(str(stereo))


def test_resample_to_16_khz() -> None:
    np = pytest.importorskip("numpy")
    from jarvis.voice.stt_worker import resample

    one_second = np.sin(np.arange(48_000) / 48_000 * 2 * np.pi * 220).astype(np.float32)
    assert len(resample(one_second, 48_000)) == 16_000
    assert len(resample(one_second[:44_100], 44_100)) == 16_000
    assert resample(one_second[:16_000], 16_000) is not None
    tone = resample(one_second, 48_000)
    assert abs(float(np.max(tone)) - 1.0) < 0.01  # a 220 Hz tone survives intact


def test_voice_benchmark_reports_accuracy_and_timing(tmp_path: Path) -> None:
    from jarvis.voice.benchmark import normalise, render, run_voice_benchmark

    def synth(sentence: str, dest: Path, voice: str | None) -> None:
        dest.write_text("Open Safari!" if "Safari" in sentence else "quit music please")

    report = run_voice_benchmark(
        fake_voice(tmp_path), tmp_path, sentences=["open Safari", "quit Music"], synth=synth
    )
    assert [(s.heard, s.exact) for s in report.samples] == [
        ("Open Safari!", True),
        ("quit music please", False),
    ]
    assert all(s.wall_seconds > 0 for s in report.samples)
    text = render(report)
    assert "Exact transcripts:      1/2" in text and "said: 'quit Music'" in text
    assert normalise("What's on the To-Do list?") == "what's on the to do list"


# -- terminal ---------------------------------------------------------------------------


def test_listen_once_stops_on_enter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def press_enter(finished: threading.Event) -> bool:
        return True

    transcript = voice_cli.listen_once(fake_voice(tmp_path), waiter=press_enter)
    assert transcript.text == "open safari" and transcript.reason == "stop"
    assert "Listening…" in capsys.readouterr().out


def test_listen_once_stops_on_its_own(tmp_path: Path) -> None:
    def wait(finished: threading.Event) -> bool:
        assert finished.wait(5)
        return False

    transcript = voice_cli.listen_once(fake_voice(tmp_path, max_seconds=0.3), waiter=wait)
    assert transcript.reason == "limit"


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["", ""], "open safari"),
        (["y"], "open safari"),
        (["n"], None),
        (["e", "open notes"], "open notes"),
    ],
)
def test_transcript_is_confirmed_before_running(answers: list[str], expected: str | None) -> None:
    replies = iter(answers)
    assert voice_cli.confirm_transcript("open safari", lambda _p: next(replies)) == expected
    assert voice_cli.confirm_transcript("", lambda _p: "y") is None


# -- service ----------------------------------------------------------------------------


class VoiceUI(ScriptedUI):
    def __init__(self, *, use: bool = True, edited: str | None = "open safari") -> None:
        super().__init__()
        self.use = use
        self.edited = edited
        self.listening: list[str] = []
        self.heard: list[str] = []

    def listen(self, title: str, message: str, finished: threading.Event) -> bool:
        self.listening.append(message)
        return self.use

    def edit(self, title: str, message: str, text: str) -> str | None:
        self.heard.append(text)
        return self.edited


def test_push_to_talk_runs_the_confirmed_transcript(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, tmp_path: Path
) -> None:
    ui = VoiceUI()
    assistant, loop, runtime = make(
        make_runtime, ui, voice=fake_voice(tmp_path), voice_shortcut="⌃⌥⌘K"
    )
    drive(assistant, loop, assistant.push_to_talk)
    assert "press ⌃⌥⌘K again" in ui.listening[0]
    assert ui.heard == ["open safari"]
    assert fake_platform.apps.launched == ["Safari"]
    assert runtime.memory is not None and runtime.memory.history()[0].source == "voice"


def test_push_to_talk_logs_decisions_without_the_words(
    make_runtime: MakeRuntime, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.runtime.assistant")
    for ui in (VoiceUI(), VoiceUI(edited="open notes"), VoiceUI(edited=None), VoiceUI(use=False)):
        assistant, loop, _ = make(make_runtime, ui, voice=fake_voice(tmp_path))
        drive(assistant, loop, assistant.push_to_talk)
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Push-to-talk")]
    assert messages == [
        "Push-to-talk: running the transcript (as heard)",
        "Push-to-talk: running the transcript (edited)",
        "Push-to-talk: transcript not run (Cancel in the edit dialog)",
        "Push-to-talk: recording discarded (Cancel in the Listening dialog)",
    ]
    assert "safari" not in caplog.text.lower() and "open notes" not in caplog.text


def test_edited_transcript_is_what_runs(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, tmp_path: Path
) -> None:
    ui = VoiceUI(edited="open notes")
    assistant, loop, _ = make(make_runtime, ui, voice=fake_voice(tmp_path))
    drive(assistant, loop, assistant.push_to_talk)
    assert fake_platform.apps.launched == ["Notes"]


@pytest.mark.parametrize("ui", [VoiceUI(use=False), VoiceUI(edited=None)])
def test_cancelling_runs_nothing(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, tmp_path: Path, ui: VoiceUI
) -> None:
    assistant, loop, _ = make(make_runtime, ui, voice=fake_voice(tmp_path))
    drive(assistant, loop, assistant.push_to_talk)
    assert fake_platform.apps.launched == []


def test_microphone_problems_are_explained(make_runtime: MakeRuntime, tmp_path: Path) -> None:
    ui = VoiceUI()
    assistant, loop, _ = make(make_runtime, ui, voice=fake_voice(tmp_path, "silent"))
    drive(assistant, loop, assistant.push_to_talk)
    assert ui.shown == ["The microphone gave only silence.\n\ngrant it"]
    assert ui.heard == []


def test_missing_model_is_explained_without_recording(
    make_runtime: MakeRuntime, tmp_path: Path
) -> None:
    ui = VoiceUI()
    voice = VoiceInput(VoiceSettings(tmp_path / "missing.bin"))
    assistant, loop, _ = make(make_runtime, ui, voice=voice)
    drive(assistant, loop, assistant.push_to_talk)
    assert ui.listening == [] and "jarvis voice download" in ui.shown[0]


def test_hotkeys_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        JarvisConfig.model_validate(
            {"voice": {"push_to_talk": True, "push_to_talk_hotkey": "ctrl+option+cmd+j"}}
        )
    assert JarvisConfig().voice.push_to_talk is False  # off by default


# -- CLI --------------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakePlatform:
    platform = FakePlatform()
    monkeypatch.setattr(cli, "get_platform", lambda: platform)
    monkeypatch.setattr(cli, "InteractiveConfirmer", lambda: ScriptedConfirmer(False))
    monkeypatch.setattr(
        "jarvis.config.default_config_file", lambda: tmp_path / "cfg" / "config.toml"
    )
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("JARVIS_AI_ENABLED", "false")
    return platform


def test_cli_listen_runs_after_confirmation(
    cli_env: FakePlatform, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    heard = Transcript("open safari", 1.5, 0.1, "silence")
    monkeypatch.setattr(voice_cli, "listen_once", lambda voice: heard)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["listen"]) == cli.EXIT_NOT_DONE
    assert cli_env.apps.launched == []
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli.main(["listen"]) == cli.EXIT_OK
    assert cli_env.apps.launched == ["Safari"]
    assert 'Heard: "open safari"' in capsys.readouterr().out


def test_cli_download_asks_first(
    cli_env: FakePlatform, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[Path] = []

    def downloader(spec: object, dest: Path) -> Path:
        calls.append(dest)
        return dest

    dest = tmp_path / DEFAULT_STT_MODEL.filename  # never the real models folder
    config = JarvisConfig.model_validate({"voice": {"stt_model_path": str(dest)}})
    declined = voice_cli.download(
        config, assume_yes=False, input_fn=lambda _p: "n", downloader=downloader
    )
    assert declined == voice_cli.EXIT_NOT_DONE
    assert voice_cli.download(config, assume_yes=True, downloader=downloader) == 0
    assert calls == [config.voice.resolved_stt_model_path()]
    other = JarvisConfig.model_validate({"voice": {"stt_model_path": str(tmp_path / "x.bin")}})
    assert voice_cli.download(other, assume_yes=True, downloader=downloader) == 1


# -- the real model (opt-in) ------------------------------------------------------------

STT_MODEL = JarvisConfig().voice.resolved_stt_model_path()


@pytest.mark.model
@pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_MODEL_TESTS") != "1", reason="set JARVIS_RUN_MODEL_TESTS=1"
)
@pytest.mark.skipif(not STT_MODEL.is_file(), reason=f"speech model not downloaded: {STT_MODEL}")
@pytest.mark.skipif(not Path("/usr/bin/afconvert").exists(), reason="needs macOS afconvert")
def test_real_model_transcribes_synthesized_speech(tmp_path: Path) -> None:
    sentences = {
        "open my Stock Price Predictor project and run it": "stock price predictor",
        "remember that my editor is Visual Studio Code": "visual studio code",
    }
    voice = VoiceInput(VoiceSettings(STT_MODEL))
    for index, (sentence, expected) in enumerate(sentences.items()):
        aiff, wav = tmp_path / f"{index}.aiff", tmp_path / f"{index}.wav"
        subprocess.run(["/usr/bin/say", "-o", str(aiff), sentence], check=True)
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
                str(aiff),
                str(wav),
            ],
            check=True,
        )
        transcript = voice.start(audio_file=wav).result()
        print(f"\n{sentence!r} -> {transcript}")
        assert expected in transcript.text.lower()
        assert transcript.transcribe_seconds < 2
