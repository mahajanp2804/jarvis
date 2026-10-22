from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis import main as cli
from jarvis.agent.orchestrator import Reply
from jarvis.agent.planner import Plan, PlanAction
from jarvis.api.schemas import ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.platforms.macos_speech import SaySpeaker, list_voices
from jarvis.voice.speech import SpeechOutput, Voice, clean_for_speech, spoken_text
from tests.fakes import FakePlatform, ScriptedConfirmer
from tests.test_assistant import ScriptedUI, drive, make

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]

VOICES = """\
Albert              en_US    # Hello! My name is Albert.
Eddy (English (US)) en_US    # Hello! My name is Eddy.
Samantha (English (US)) en_US    # Hello! My name is Samantha.
Amélie              fr_CA    # Bonjour! Je m'appelle Amélie.
"""


class FakeSpeaker(SpeechOutput):
    def __init__(self) -> None:
        self.voice: str | None = None
        self.said: list[str] = []
        self.stops = 0
        self.waited = False

    def speak(self, text: str) -> None:
        self.said.append(text)

    def stop(self) -> None:
        self.stops += 1

    @property
    def speaking(self) -> bool:
        return False

    def wait(self, timeout: float | None = None) -> bool:
        self.waited = True
        return True

    def voices(self) -> list[Voice]:
        return [Voice("Samantha", "en_US"), Voice("Amélie", "fr_CA")]


def fake_say(tmp_path: Path, *, sleep: float = 0) -> Path:
    """A stand-in for /usr/bin/say that records its arguments and stdin."""
    script = tmp_path / "say"
    log = tmp_path / "say.log"
    script.write_text(
        "#!/bin/sh\n"
        f'if [ "$2" = "?" ]; then cat <<"EOF"\n{VOICES}EOF\nexit 0; fi\n'
        f'echo "args: $*" >> "{log}"\n'
        f'printf "text: %s\\n" "$(cat)" >> "{log}"\n'
        f"sleep {sleep}\n"
    )
    script.chmod(0o755)
    return script


def result(message: str, status: ToolStatus = ToolStatus.SUCCESS) -> ToolResult:
    return ToolResult(tool="apps.open", status=status, message=message)


# -- what is said ----------------------------------------------------------------------


def test_clean_for_speech() -> None:
    assert clean_for_speech("✓ Opened Safari.\n  details") == "Opened Safari."
    assert (
        clean_for_speech("Started `python3 app.py` at http://127.0.0.1:8778/docs?x=1")
        == "Started python3 app.py at 127.0.0.1 port 8778"
    )
    assert clean_for_speech("Read ~/Desktop/todo.txt") == "Read Desktop/todo.txt"
    long = "First sentence here. " + "word " * 100
    assert clean_for_speech(long, limit=60) == "First sentence here."
    assert clean_for_speech("x" * 30 + " " + "y" * 50, limit=60).endswith("…")


def test_spoken_text_picks_the_answer() -> None:
    plan = Plan(PlanAction.RESPOND, "It lists two tasks.", (), 1, ())
    reply = Reply("It lists two tasks.\nSteps:\n  ✓ Read ~/Desktop/todo.txt", plan=plan)
    assert spoken_text(reply) == "It lists two tasks."

    steps = Reply("✓ Found it\nStarted the server.", [result("Found it"), result("Started it.")])
    assert spoken_text(steps) == "Started it."

    stopped = Reply(
        "...",
        [result("Quit Notes?", ToolStatus.CANCELLED)],
        stopped="You declined, so I stopped.",
    )
    assert spoken_text(stopped) == "Quit Notes? You declined, so I stopped."
    assert spoken_text(Reply("I can't do that yet.", handled=False)) == "I can't do that yet."


# -- the macOS speaker --------------------------------------------------------------------


def test_say_gets_text_on_stdin_and_options(tmp_path: Path) -> None:
    speaker = SaySpeaker("Samantha", 180, say=fake_say(tmp_path))
    speaker.speak("--help me; rm -rf ~")
    assert speaker.wait(timeout=5)
    log = (tmp_path / "say.log").read_text()
    assert "args: -v Samantha -r 180 -f -" in log
    assert "text: --help me; rm -rf ~" in log  # never an argument


def test_new_speech_and_stop_interrupt(tmp_path: Path) -> None:
    speaker = SaySpeaker(say=fake_say(tmp_path, sleep=30))
    speaker.speak("first")
    time.sleep(0.2)
    assert speaker.speaking
    started = time.monotonic()
    speaker.speak("second")  # stops "first"
    speaker.stop()
    assert time.monotonic() - started < 3
    assert not speaker.speaking
    speaker.speak("   ")  # nothing to say: stays silent
    assert not speaker.speaking


def test_voices_and_check(tmp_path: Path) -> None:
    say = fake_say(tmp_path)
    names = [(v.name, v.language) for v in list_voices(say)]
    assert names == [
        ("Albert", "en_US"),
        ("Eddy (English (US))", "en_US"),
        ("Samantha (English (US))", "en_US"),
        ("Amélie", "fr_CA"),
    ]
    SaySpeaker("samantha", say=say).check()
    with pytest.raises(JarvisError, match="isn't installed"):
        SaySpeaker("Nobody", say=say).check()
    with pytest.raises(JarvisError, match="missing"):
        SaySpeaker(say=tmp_path / "nope").check()


def test_rate_must_be_sensible() -> None:
    with pytest.raises(ValueError, match="80-500"):
        JarvisConfig.model_validate({"voice": {"rate": 40}})
    assert JarvisConfig().voice.speak_replies is False  # off by default


# -- the service ------------------------------------------------------------------------


def test_service_speaks_replies_and_the_shortcut_stops_them(make_runtime: MakeRuntime) -> None:
    speaker = FakeSpeaker()
    ui = ScriptedUI("list apps matching photo")
    assistant, loop, _ = make(make_runtime, ui, speaker=speaker)
    drive(assistant, loop)
    assert len(speaker.said) == 1 and speaker.said[0].startswith("Found")
    assert speaker.stops >= 2  # when the shortcut was pressed, and at shutdown


def test_requests_from_the_socket_are_not_spoken(make_runtime: MakeRuntime) -> None:
    speaker = FakeSpeaker()
    assistant, loop, _ = make(make_runtime, ScriptedUI(), speaker=speaker)
    drive(assistant, loop, lambda: assistant.submit("open safari", interactive=False))
    assert speaker.said == []


# -- CLI --------------------------------------------------------------------------------


@pytest.fixture
def cli_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSpeaker:
    speaker = FakeSpeaker()
    platform = FakePlatform()

    def speech_output(voice: str | None = None, rate: int | None = None) -> FakeSpeaker:
        speaker.voice = voice
        return speaker

    monkeypatch.setattr(platform, "speech_output", speech_output)
    monkeypatch.setattr(cli, "get_platform", lambda: platform)
    monkeypatch.setattr(cli, "InteractiveConfirmer", lambda: ScriptedConfirmer(False))
    monkeypatch.setattr(
        "jarvis.config.default_config_file", lambda: tmp_path / "cfg" / "config.toml"
    )
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("JARVIS_AI_ENABLED", "false")
    return speaker


def test_run_speak(cli_platform: FakeSpeaker, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--speak", "open safari"]) == cli.EXIT_OK
    assert cli_platform.said == ["Opened Safari."] and cli_platform.waited
    cli_platform.said.clear()
    assert cli.main(["run", "open safari"]) == cli.EXIT_OK
    assert cli_platform.said == []  # not without --speak or [voice] speak_replies


def test_voice_commands(cli_platform: FakeSpeaker, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["voice", "voices", "--language", "fr"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Amélie" in out and "Samantha" not in out
    assert cli.main(["voice", "say", "hello", "there"]) == cli.EXIT_OK
    assert cli_platform.said == ["hello there"] and cli_platform.voice is None
    assert cli.main(["voice", "say", "--voice", "Tara", "hi"]) == cli.EXIT_OK
    assert cli_platform.said[-1] == "hi" and cli_platform.voice == "Tara"
