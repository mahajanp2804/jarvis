"""Shared fixtures: build a runtime on the in-memory fake platform."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from jarvis.bootstrap import Runtime, build_runtime
from jarvis.config import JarvisConfig
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer


@pytest.fixture
def fake_platform() -> FakePlatform:
    return FakePlatform()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_runtime(
    tmp_path: Path, fake_platform: FakePlatform, fake_llm: FakeLLM
) -> Iterator[Callable[..., tuple[Runtime, ScriptedConfirmer]]]:
    """Factory: build a runtime on the fake platform with logs in tmp_path."""
    runtimes: list[Runtime] = []

    def factory(
        *,
        confirm: bool = True,
        dry_run: bool = False,
        policy: dict[str, object] | None = None,
        applications: dict[str, object] | None = None,
        ai: dict[str, object] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Runtime, ScriptedConfirmer]:
        config = JarvisConfig.model_validate(
            {
                "general": {"dry_run": dry_run, "log_dir": str(tmp_path / "logs")},
                "applications": {
                    "launch_timeout_seconds": 0.3,
                    "quit_timeout_seconds": 0.3,
                    **(applications or {}),
                },
                "policy": policy or {},
                "ai": {"idle_unload_seconds": 0, **(ai or {})},
            }
        )
        confirmer = ScriptedConfirmer(confirm)
        runtime = build_runtime(
            config,
            fake_platform,
            confirmer,
            cancel_event=cancel_event,
            poll_interval=0.01,
            backend=fake_llm,
        )
        runtimes.append(runtime)
        return runtime, confirmer

    yield factory
    for runtime in runtimes:
        runtime.close()
