from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from jarvis import service_cli
from jarvis.config import JarvisConfig
from jarvis.platforms.base import ProcessStats
from tests.fakes import FakePlatform


def test_service_program_uses_this_python_and_absolute_config(tmp_path: Path) -> None:
    assert service_cli.service_program(None) == [sys.executable, "-m", "jarvis", "daemon"]
    program = service_cli.service_program(Path("relative.toml"))
    assert program[-2] == "--config" and Path(program[-1]).is_absolute()


class StatsPlatform(FakePlatform):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process_stats(self, pid: int | None = None) -> ProcessStats | None:
        self.calls += 1
        later = self.calls > 2  # two pids sampled before, two after
        return ProcessStats(
            rss_bytes=50 * 2**20,
            footprint_bytes=30 * 2**20,
            wakeups=100 if later else 0,
            cpu_seconds=0.2 if later else 0.0,
        )


def test_measure_reports_rates_for_service_and_worker(capsys: pytest.CaptureFixture[str]) -> None:
    platform = StatsPlatform()
    code = service_cli.measure(
        platform,
        seconds=10,
        sleep=lambda _s: None,
        query=lambda: {"pid": 100, "worker_pids": [101], "model_loaded": True},
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "[100, 101]" in out
    assert "CPU        4.000%" in out  # (0.2+0.2 CPU-s) / 10 s
    assert "Wakeups    20.00/s" in out
    assert "Footprint  60.0 MB" in out


def test_measure_when_not_running(capsys: pytest.CaptureFixture[str]) -> None:
    assert service_cli.measure(FakePlatform(), seconds=1, query=lambda: None) == 3
    assert "isn't running" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ({"ok": True, "text": "Opened Safari.", "handled": True, "statuses": ["success"]}, 0),
        ({"ok": True, "text": "Which app?", "handled": False, "statuses": []}, 3),
        ({"ok": True, "text": "x", "handled": True, "statuses": ["failed"]}, 1),
        ({"ok": True, "text": "x", "handled": True, "statuses": ["cancelled"]}, 3),
        ({"ok": False, "error": "busy"}, 1),
    ],
)
def test_ask_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reply: dict[str, Any],
    code: int,
) -> None:
    sent: list[dict[str, Any]] = []

    def fake_send(path: Path, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        sent.append(request)
        return reply

    monkeypatch.setattr(service_cli, "send_request", fake_send)
    assert service_cli.ask("open safari", JarvisConfig()) == code
    assert sent == [{"cmd": "ask", "text": "open safari"}]


def test_ask_without_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from jarvis.runtime.lifecycle import DaemonPaths

    monkeypatch.setattr(
        DaemonPaths, "default", classmethod(lambda cls: cls(tmp_path / "l", tmp_path / "s.sock"))
    )
    assert service_cli.ask("hi", JarvisConfig()) == 1
    assert service_cli.activate() == 1


def test_service_commands_unsupported_platform(capsys: pytest.CaptureFixture[str]) -> None:
    code = service_cli.service_command("install", JarvisConfig(), FakePlatform(), config_path=None)
    assert code == 1
    assert "isn't implemented" in capsys.readouterr().err
