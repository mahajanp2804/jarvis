from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

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
    assert "powermetrics" in text

    out = tmp_path / "bench.json"
    report.write_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["requests"][0]["tools"] == ["apps.open"]
    assert "generation_tokens_per_second" in data["requests"][0]
