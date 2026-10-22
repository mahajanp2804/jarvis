from __future__ import annotations

from pathlib import Path

import pytest

from jarvis import __version__
from jarvis import main as cli
from jarvis.ai.registry import DEFAULT_MODEL
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer, plan_json


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr("jarvis.bootstrap.build_backend", lambda ai: fake)
    return fake


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: FakeLLM) -> FakePlatform:
    """Isolate the CLI: fake OS + model, temp config/log/model paths, scripted confirmations."""
    fake = FakePlatform()
    monkeypatch.setattr(cli, "get_platform", lambda: fake)
    monkeypatch.setattr(cli, "InteractiveConfirmer", lambda: ScriptedConfirmer(False))
    monkeypatch.setattr(
        "jarvis.config.default_config_file", lambda: tmp_path / "cfg" / "config.toml"
    )
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    monkeypatch.delenv("JARVIS_DRY_RUN", raising=False)
    monkeypatch.delenv("JARVIS_AI_ENABLED", raising=False)
    monkeypatch.setenv("JARVIS_MODEL_PATH", str(tmp_path / "models" / DEFAULT_MODEL.filename))
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "logs"))
    return fake


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert f"jarvis {__version__}" in capsys.readouterr().out


def test_run_list(platform: FakePlatform, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "list", "apps"]) == cli.EXIT_OK
    assert "Safari" in capsys.readouterr().out


def test_run_open(platform: FakePlatform, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "open safari"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "Opened Safari."
    assert platform.apps.launched == ["Safari"]


@pytest.mark.parametrize(
    "argv",
    [["--dry-run", "run", "open Safari"], ["run", "--dry-run", "open Safari"]],
)
def test_dry_run_flag_before_or_after_command(
    platform: FakePlatform, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert cli.main(argv) == cli.EXIT_OK
    assert "[dry run] Would: Open Safari" in capsys.readouterr().out
    assert platform.apps.launched == []


def test_dry_run_from_environment(
    platform: FakePlatform, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JARVIS_DRY_RUN", "1")
    assert cli.main(["apps", "open", "Safari"]) == cli.EXIT_OK
    assert platform.apps.launched == []


def test_ai_plan_is_executed(
    platform: FakePlatform, llm: FakeLLM, capsys: pytest.CaptureFixture[str]
) -> None:
    llm.replies.append(
        plan_json(
            "execute",
            "Opening both.",
            ("apps.open", {"name": "Safari"}),
            ("apps.open", {"name": "Notes"}),
        )
    )
    assert cli.main(["run", "open my browser and the notes app"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "Opened Safari." in captured.out
    assert "Opened Notes." in captured.out
    assert "thinking" in captured.err
    assert platform.apps.launched == ["Safari", "Notes"]


def test_ai_clarification_exit_code(
    platform: FakePlatform, llm: FakeLLM, capsys: pytest.CaptureFixture[str]
) -> None:
    llm.replies.append(plan_json("clarify", "Which project should I open?"))
    code = cli.main(["run", "open my Stock Price Predictor project and run it"])
    assert code == cli.EXIT_NOT_DONE
    assert "Which project" in capsys.readouterr().out


def test_ai_disabled(
    platform: FakePlatform,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("JARVIS_AI_ENABLED", "false")
    assert cli.main(["run", "open my browser"]) == cli.EXIT_NOT_DONE
    assert "disabled" in capsys.readouterr().out
    assert llm.loads == 0


def test_missing_model_gives_actionable_message(
    platform: FakePlatform, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jarvis.ai.llama_cpp_backend import LlamaCppBackend

    monkeypatch.setattr(
        "jarvis.bootstrap.build_backend", lambda ai: LlamaCppBackend(ai.resolved_model_path())
    )
    assert cli.main(["run", "open my browser"]) == cli.EXIT_NOT_DONE
    out = capsys.readouterr().out
    assert "Model file not found" in out
    assert "jarvis model download" in out


def test_rule_commands_never_load_the_model(platform: FakePlatform, llm: FakeLLM) -> None:
    assert cli.main(["run", "open Safari"]) == cli.EXIT_OK
    assert llm.loads == 0


def test_declined_quit_exit_code(
    platform: FakePlatform, capsys: pytest.CaptureFixture[str]
) -> None:
    platform.apps.running.add("Notes")
    assert cli.main(["apps", "quit", "Notes"]) == cli.EXIT_NOT_DONE
    assert "Cancelled" in capsys.readouterr().out
    assert platform.apps.quit_requests == []


def test_not_found_exit_code(platform: FakePlatform, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["apps", "open", "notepad"]) == cli.EXIT_ERROR
    assert "Did you mean: Notes?" in capsys.readouterr().out
    assert cli.main(["apps", "open", "safarii"]) == cli.EXIT_NOT_DONE  # asks, never guesses
    out = capsys.readouterr().out
    assert "Did you mean Safari?" in out and "Which one" not in out
    assert platform.apps.launched == []


def test_audit_log_written(platform: FakePlatform, tmp_path: Path) -> None:
    cli.main(["run", "open Safari"])
    assert (tmp_path / "logs" / "audit.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_config_init_show_and_path(
    platform: FakePlatform, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "cfg" / "config.toml"
    assert cli.main(["config", "path"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == str(target)

    assert cli.main(["config", "init"]) == cli.EXIT_OK
    assert target.is_file()
    assert cli.main(["config", "init"]) == cli.EXIT_ERROR
    assert "already exists" in capsys.readouterr().err

    assert cli.main(["config", "show"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert '"confirm_at_or_above": "medium"' in out


def test_invalid_config_is_reported(
    platform: FakePlatform, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[policy]\nconfirm_at_or_above = 'never'\n", encoding="utf-8")
    assert cli.main(["--config", str(bad), "run", "list apps"]) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "Invalid configuration" in err
    assert "confirm_at_or_above" in err


def test_doctor(platform: FakePlatform, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["doctor"]) == cli.EXIT_ERROR  # the model file is missing
    out = capsys.readouterr().out
    assert "✓ Fake platform: ok" in out
    assert "Application discovery: 7 apps" in out
    assert "✗ Model file" in out


def test_shell_session(
    platform: FakePlatform, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["help", "open notes", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert cli.main(["shell"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "open <app>" in out
    assert "Opened Notes." in out


def test_shell_eof_exits(platform: FakePlatform, monkeypatch: pytest.MonkeyPatch) -> None:
    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert cli.main(["shell"]) == cli.EXIT_OK


def test_unknown_command_is_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["fly"])
    assert info.value.code == 2
