from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from jarvis import model_cli
from jarvis.ai.registry import DEFAULT_MODEL
from jarvis.config import JarvisConfig


def config_with_model(path: Path) -> JarvisConfig:
    return JarvisConfig.model_validate({"ai": {"model_path": str(path)}})


def test_info(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert model_cli.model_info(config_with_model(tmp_path / DEFAULT_MODEL.filename)) == 0
    out = capsys.readouterr().out
    assert "Qwen3-4B-Instruct-2507" in out
    assert "Present" in out and "no" in out
    assert "300s" in out


def test_check_missing_and_present_custom(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert model_cli.model_check(config_with_model(tmp_path / "x.gguf"), verify=False) == 1
    assert "not found" in capsys.readouterr().out
    custom = tmp_path / "custom.gguf"
    custom.write_bytes(b"GGUF" + b"\0" * 10)
    assert model_cli.model_check(config_with_model(custom), verify=True) == 0
    assert "can't be verified" in capsys.readouterr().out


def test_download_refuses_custom_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = model_cli.model_download(
        config_with_model(tmp_path / "other.gguf"), assume_yes=True, downloader=pytest.fail
    )
    assert code == 1
    assert "not the registry model" in capsys.readouterr().out


def test_download_requires_explicit_yes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = config_with_model(tmp_path / DEFAULT_MODEL.filename)
    code = model_cli.model_download(
        config, assume_yes=False, input_fn=lambda _p: "", downloader=pytest.fail
    )
    assert code == 3
    out = capsys.readouterr().out
    assert "2.50 GB" in out and "Apache-2.0" in out and "Not downloaded" in out


def test_download_after_confirmation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / DEFAULT_MODEL.filename
    calls: list[tuple[Any, Path]] = []

    def fake_downloader(spec: Any, path: Path, **kwargs: Any) -> Path:
        calls.append((spec, path))
        return path

    code = model_cli.model_download(
        config_with_model(dest),
        assume_yes=False,
        input_fn=lambda _p: "y",
        downloader=fake_downloader,
    )
    assert code == 0
    assert calls == [(DEFAULT_MODEL, dest)]


def test_download_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = b"GGUF" + b"\1" * 100
    dest = tmp_path / DEFAULT_MODEL.filename
    dest.write_bytes(payload)
    monkeypatch.setattr(
        model_cli,
        "DEFAULT_MODEL",
        DEFAULT_MODEL.__class__(
            **{
                **DEFAULT_MODEL.__dict__,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
    )
    monkeypatch.setattr(model_cli, "spec_for_filename", lambda _n: model_cli.DEFAULT_MODEL)
    code = model_cli.model_download(
        config_with_model(dest), assume_yes=True, downloader=pytest.fail
    )
    assert code == 0
    assert "already at" in capsys.readouterr().out
