"""End-to-end tests with the real local model on the *fake* app platform.

Opt in with ``JARVIS_RUN_MODEL_TESTS=1`` (they load ~3 GB and take ~20 s).
The real model plans; the in-memory platform executes, so no real
application is ever touched. Assertions check tool choices, not wording.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from jarvis.agent.planner import PlanAction
from jarvis.ai.model_manager import ModelManager
from jarvis.bootstrap import Runtime, build_backend, build_runtime
from jarvis.config import JarvisConfig
from tests.fakes import FakePlatform, ScriptedConfirmer

MODEL = JarvisConfig().ai.resolved_model_path()

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("JARVIS_RUN_MODEL_TESTS") != "1",
        reason="set JARVIS_RUN_MODEL_TESTS=1 to run real-model tests",
    ),
    pytest.mark.skipif(not MODEL.is_file(), reason=f"model not downloaded: {MODEL}"),
]


@pytest.fixture(scope="module")
def platform() -> FakePlatform:
    return FakePlatform()


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory, platform: FakePlatform) -> Iterator[Runtime]:
    config = JarvisConfig.model_validate(
        {
            "general": {"log_dir": str(tmp_path_factory.mktemp("logs"))},
            "applications": {"launch_timeout_seconds": 0.3, "quit_timeout_seconds": 0.3},
            "ai": {"idle_unload_seconds": 3600},
        }
    )
    rt = build_runtime(config, platform, ScriptedConfirmer(True), poll_interval=0.01)
    yield rt
    rt.close()


def plan_tools(runtime: Runtime, request: str) -> tuple[PlanAction, list[tuple[str, object]]]:
    assert runtime.planner is not None
    plan = runtime.planner.plan(request)
    return plan.action, [
        (s.tool, s.arguments.get("name") or s.arguments.get("query")) for s in plan.steps
    ]


def test_maps_generic_names_to_installed_apps(runtime: Runtime) -> None:
    action, steps = plan_tools(runtime, "open my web browser and the notes app")
    assert action is PlanAction.EXECUTE
    assert steps == [("apps.open", "Safari"), ("apps.open", "Notes")]


def test_search_request_uses_list_tool(runtime: Runtime) -> None:
    action, steps = plan_tools(runtime, "which apps do I have for editing photos?")
    assert action is PlanAction.EXECUTE
    assert len(steps) == 1 and steps[0][0] == "apps.list"
    assert "photo" in str(steps[0][1]).lower()


def test_unsupported_request_is_declined(runtime: Runtime) -> None:
    action, steps = plan_tools(runtime, "please delete everything in my downloads folder")
    assert action is PlanAction.RESPOND
    assert steps == []


def test_vague_request_asks_for_clarification(runtime: Runtime) -> None:
    action, steps = plan_tools(runtime, "open it")
    assert action is PlanAction.CLARIFY
    assert steps == []


def test_follow_up_uses_conversation_history(runtime: Runtime, platform: FakePlatform) -> None:
    runtime.orchestrator.clear_history()
    runtime.orchestrator.handle("open notes")  # rule-based
    reply = runtime.orchestrator.handle("now quit it")
    assert [r.tool for r in reply.results] == ["apps.quit"]
    assert platform.apps.quit_requests[-1] == "Notes"


def test_real_backend_unloads_after_idle_timeout(tmp_path: Path) -> None:
    manager = ModelManager(build_backend(JarvisConfig().ai), idle_unload_seconds=0.5)
    with manager.session() as backend:
        assert backend.is_loaded
    deadline = time.monotonic() + 10
    while manager.is_loaded and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not manager.is_loaded


def test_worker_process_returns_memory_and_can_be_aborted() -> None:
    import threading

    from jarvis.ai.subprocess_backend import SubprocessBackend
    from jarvis.platforms.macos import read_process_stats

    ai = JarvisConfig().ai
    backend = build_backend(ai)
    assert isinstance(backend, SubprocessBackend)
    manager = ModelManager(backend, idle_unload_seconds=3600)
    with manager.session():
        pid = backend.process_ids()[0]
        stats = read_process_stats(pid)
        assert stats is not None and stats.footprint_bytes and stats.footprint_bytes > 300 * 2**20
    manager.unload()
    assert read_process_stats(pid) is None  # worker gone: its memory is back with the OS

    # Abort while a request is generating.
    long_prompt = [{"role": "user", "content": "count to 300 in words " * 20}]
    errors: list[Exception] = []
    with manager.session() as llm:

        def generate() -> None:
            try:
                llm.complete_json(long_prompt, {"type": "string"}, max_tokens=2000, temperature=0)  # type: ignore[arg-type]
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=generate)
        thread.start()
        time.sleep(1.5)
        started = time.monotonic()
        manager.abort()
        thread.join(timeout=10)
        assert time.monotonic() - started < 2
    assert errors and "stopped" in str(errors[0])
    assert not manager.is_loaded
