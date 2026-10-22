from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jarvis.api.schemas import RiskLevel
from jarvis.config import (
    EXAMPLE_CONFIG_TOML,
    JarvisConfig,
    load_config,
    resolve_config_path,
    write_example_config,
)
from jarvis.errors import ConfigError


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults() -> None:
    config = JarvisConfig()
    assert config.general.dry_run is False
    assert config.general.log_level == "INFO"
    assert config.policy.confirm_at_or_above is RiskLevel.MEDIUM
    assert config.policy.enabled_tools == [
        "apps.*",
        "files.*",
        "projects.*",
        "terminal.*",
        "browser.*",
        "memory.*",
    ]


def test_missing_default_file_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("jarvis.config.default_config_file", lambda: tmp_path / "nope.toml")
    assert load_config(env={}) == JarvisConfig()


def test_example_config_parses_to_defaults() -> None:
    assert JarvisConfig.model_validate(tomllib.loads(EXAMPLE_CONFIG_TOML)) == JarvisConfig()


def test_loads_values_from_file(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [general]
        dry_run = true
        log_level = "debug"
        log_dir = "~/jarvis-logs"

        [applications.aliases]
        "My  Browser" = "Safari"

        [policy]
        confirm_at_or_above = "low"
        app_blocklist = ["Terminal"]
        """,
    )
    config = load_config(path, env={})
    assert config.general.dry_run is True
    assert config.general.log_level == "DEBUG"
    assert config.general.log_dir == Path("~/jarvis-logs").expanduser()
    assert config.applications.aliases == {"my browser": "Safari"}
    assert config.policy.confirm_at_or_above is RiskLevel.LOW
    assert config.policy.app_blocklist == ["Terminal"]


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "[policy]\nconfirm_above = 'low'\n")
    with pytest.raises(ConfigError, match=r"policy\.confirm_above"):
        load_config(path, env={})


def test_cannot_disable_confirmation_for_critical_actions(tmp_path: Path) -> None:
    path = write(tmp_path, "[policy]\nconfirm_at_or_above = 'critical'\n")
    with pytest.raises(ConfigError, match="always require confirmation"):
        load_config(path, env={})


def test_invalid_risk_name(tmp_path: Path) -> None:
    path = write(tmp_path, "[policy]\nconfirm_at_or_above = 'sometimes'\n")
    with pytest.raises(ConfigError, match="unknown risk level"):
        load_config(path, env={})


def test_invalid_toml_reports_file(tmp_path: Path) -> None:
    path = write(tmp_path, "[general\n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path, env={})


def test_explicit_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found") as info:
        load_config(tmp_path / "missing.toml", env={})
    assert info.value.hint and "config init" in info.value.hint


def test_env_overrides_file(tmp_path: Path) -> None:
    path = write(tmp_path, "[general]\ndry_run = false\nlog_level = 'INFO'\n")
    env = {"JARVIS_DRY_RUN": "yes", "JARVIS_LOG_LEVEL": "warning", "JARVIS_LOG_DIR": str(tmp_path)}
    config = load_config(path, env=env)
    assert config.general.dry_run is True
    assert config.general.log_level == "WARNING"
    assert config.general.resolved_log_dir() == tmp_path


def test_env_config_path(tmp_path: Path) -> None:
    path = write(tmp_path, "[general]\ndry_run = true\n")
    assert resolve_config_path(None, {"JARVIS_CONFIG": str(path)}) == (path, True)
    assert load_config(env={"JARVIS_CONFIG": str(path)}).general.dry_run is True


def test_invalid_env_bool(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="JARVIS_DRY_RUN"):
        load_config(write(tmp_path, ""), env={"JARVIS_DRY_RUN": "maybe"})


def test_timeouts_are_bounded(tmp_path: Path) -> None:
    path = write(tmp_path, "[applications]\nlaunch_timeout_seconds = 0\n")
    with pytest.raises(ConfigError, match="launch_timeout_seconds"):
        load_config(path, env={})


def test_write_example_config_does_not_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "config.toml"
    write_example_config(target)
    assert target.read_text(encoding="utf-8") == EXAMPLE_CONFIG_TOML
    target.write_text("# mine\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="already exists"):
        write_example_config(target)
    assert target.read_text(encoding="utf-8") == "# mine\n"
    write_example_config(target, overwrite=True)
    assert target.read_text(encoding="utf-8") == EXAMPLE_CONFIG_TOML


def test_ai_defaults_point_at_the_default_model() -> None:
    from jarvis.ai.registry import DEFAULT_MODEL
    from jarvis.paths import default_models_dir

    ai = JarvisConfig().ai
    assert ai.enabled is True
    assert ai.resolved_model_path() == default_models_dir() / DEFAULT_MODEL.filename
    assert ai.context_length == 4096
    assert ai.idle_unload_seconds == 300.0
    assert ai.metal_residency_sets is False


def test_ai_env_overrides(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    config = load_config(
        write(tmp_path, "[ai]\nenabled = true\n"),
        env={"JARVIS_AI_ENABLED": "0", "JARVIS_MODEL_PATH": str(model)},
    )
    assert config.ai.enabled is False
    assert config.ai.resolved_model_path() == model


@pytest.mark.parametrize(
    "line",
    ["context_length = 100", "gpu_layers = -2", "temperature = 5", "backend = 'ollama'"],
)
def test_ai_values_are_validated(tmp_path: Path, line: str) -> None:
    with pytest.raises(ConfigError, match="ai"):
        load_config(write(tmp_path, f"[ai]\n{line}\n"), env={})


def test_service_defaults_and_hotkey_normalisation(tmp_path: Path) -> None:
    assert JarvisConfig().service.hotkey == "ctrl+option+cmd+j"
    config = load_config(write(tmp_path, "[service]\nhotkey = '⌘⇧ Space'\n"), env={})
    assert config.service.hotkey == "shift+cmd+space"


@pytest.mark.parametrize(
    "line", ["hotkey = 'option+space'", "hotkey = 'cmd+'", "label = 'bad label!'"]
)
def test_service_values_are_validated(tmp_path: Path, line: str) -> None:
    with pytest.raises(ConfigError, match="service"):
        load_config(write(tmp_path, f"[service]\n{line}\n"), env={})
