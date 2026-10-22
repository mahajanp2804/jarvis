from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jarvis.ai.base import Completion
from jarvis.benchmark import render, run_benchmark
from jarvis.bootstrap import Runtime
from jarvis.platforms.base import ProcessStats
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer, plan_json

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def test_benchmark_measures_without_executing_anything(
    make_runtime: MakeRuntime,
    fake_llm: FakeLLM,
    fake_platform: FakePlatform,
    tmp_path: Path,
) -> None:
    fake_llm.replies.extend(
        [
            plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})),
            plan_json("respond", "Can't do that."),
        ]
        * 2
        + [plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"}))]
    )
    runtime, confirmer = make_runtime(dry_run=True, ai={"idle_unload_seconds": 3600})
    assert runtime.models is not None and runtime.planner is not None

    ticks = iter(range(10_000))
    stats_calls = iter(range(10_000))

    def stats() -> ProcessStats:
        n = next(stats_calls)
        return ProcessStats(rss_bytes=(100 + n) * 2**20, footprint_bytes=50 * 2**20, wakeups=n)

    slept: list[float] = []
    report = run_benchmark(
        runtime.models,
        runtime.planner,
        runtime.orchestrator,
        stats,
        prompts=["open my browser", "delete my files"],
        repeats=2,
        idle_seconds=5,
        clock=lambda: float(next(ticks)),
        cpu_clock=lambda: 0.0,
        sleep=slept.append,
    )

    assert slept == [5, 5]
    assert report.idle_unloaded.cpu_percent == 0.0
    assert report.idle_unloaded.wakeups_per_second is not None
    assert [r.action for r in report.requests] == ["execute", "respond", "execute", "respond"]
    assert report.requests[0].tools == ["apps.open"]
    assert report.unload is not None and report.reload is not None
    assert report.after_reload is not None and report.after_reload.action == "execute"
    assert fake_llm.loads == 2  # initial load + reload
    assert not fake_llm.loaded  # left unloaded
    # Nothing was executed: planning only, and the fast-path probe was a dry run.
    assert fake_platform.apps.launched == []
    assert confirmer.prompts == []

    text = render(report)
    assert "model unloaded" in text and "model loaded" in text
    assert "first request after reload" in text
    assert "prompt cache on disk" in text and "off" in text
    assert "powermetrics" in text

    out = tmp_path / "bench.json"
    report.write_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["requests"][0]["tools"] == ["apps.open"]
    assert "generation_tokens_per_second" in data["requests"][0]


def test_benchmark_reports_the_disk_prompt_cache(
    make_runtime: MakeRuntime,
    fake_llm: FakeLLM,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = fake_llm.complete_json
    seen_loads = 0

    def first_after_load_uses_disk(*args: Any, **kwargs: Any) -> Completion:
        nonlocal seen_loads
        completion = complete(*args, **kwargs)
        if fake_llm.loads == seen_loads:
            return completion
        seen_loads = fake_llm.loads
        event = "saved" if seen_loads == 1 else "restored"
        return dataclasses.replace(
            completion, prompt_cache=event, prompt_cache_seconds=0.25, cached_prompt_tokens=1749
        )

    monkeypatch.setattr(fake_llm, "complete_json", first_after_load_uses_disk)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "k.0000abcd.llstate").write_bytes(b"x" * 2 * 2**20)
    runtime, _ = make_runtime(dry_run=True, ai={"idle_unload_seconds": 3600})
    assert runtime.models is not None and runtime.planner is not None

    report = run_benchmark(
        runtime.models,
        runtime.planner,
        runtime.orchestrator,
        lambda: ProcessStats(rss_bytes=100 * 2**20, footprint_bytes=50 * 2**20, wakeups=0),
        prompts=["open my browser", "delete my files"],
        repeats=1,
        idle_seconds=0,
        prompt_cache_dir=cache_dir,
        sleep=lambda _s: None,
    )

    assert [r.prompt_cache for r in report.requests] == ["saved", None]
    assert report.after_reload is not None and report.after_reload.prompt_cache == "restored"
    assert report.after_reload.memory_after is not None
    assert (report.prompt_cache_files, report.prompt_cache_disk_mb) == (1, 2.0)
    text = render(report)
    assert "then saved to disk in 0.25s" in text
    assert "(1749 prompt tokens restored from disk in 0.25s); RSS 100 MB" in text
    assert "2 MB in 1 file(s)" in text
