"""Benchmark the cost of the local model: load time, inference, memory, idle CPU.

What is measured (all in this process):

* **Idle cost** with the model unloaded and loaded: CPU % and wakeups/second
  while Jarvis waits for input. Wakeups are what macOS's Energy Impact is
  largely based on.
* **Load / unload / reload**: wall time, CPU time and memory after each.
  The reload usually hits the OS file cache, so it is faster than a cold load
  after a reboot.
* **Planning requests**: end-to-end latency, prompt/generation split, tokens,
  and how many prompt tokens were reused from llama.cpp's cache.
* **Rule-based fast path** for comparison (no model involved).

Limits, stated honestly: CPU time does not include GPU (Metal) work, and battery
drain is not measured directly. For package power use
``sudo powermetrics --samplers cpu_power,gpu_power -i 1000`` while the benchmark
runs. The benchmark only *plans*; it never executes a tool.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jarvis.agent.orchestrator import Orchestrator
from jarvis.agent.planner import Planner
from jarvis.ai.model_manager import ModelManager
from jarvis.platforms.base import PlatformAdapter, ProcessStats

DEFAULT_PROMPTS = [
    "open my web browser and the notes app",
    "which apps do I have for editing photos?",
    "quit the music player",
    "please delete everything in my downloads folder",
]

_MB = 1024 * 1024


@dataclass
class Snapshot:
    rss_mb: float | None
    footprint_mb: float | None
    wakeups: int | None

    @classmethod
    def take(cls, stats: Callable[[], ProcessStats | None]) -> Snapshot:
        s = stats()
        if s is None:
            return cls(None, None, None)
        footprint = s.footprint_bytes / _MB if s.footprint_bytes is not None else None
        return cls(round(s.rss_bytes / _MB, 1), _round(footprint), s.wakeups)


@dataclass
class IdleSample:
    seconds: float
    cpu_percent: float
    wakeups_per_second: float | None
    memory: Snapshot


@dataclass
class TimedStep:
    wall_seconds: float
    cpu_seconds: float
    memory_after: Snapshot


@dataclass
class RequestSample:
    prompt: str
    wall_seconds: float
    cpu_seconds: float
    action: str
    tools: list[str]
    attempts: int
    prompt_tokens: int
    cached_prompt_tokens: int | None
    completion_tokens: int
    prompt_eval_seconds: float | None
    generation_seconds: float | None

    @property
    def generation_tokens_per_second(self) -> float | None:
        if not self.generation_seconds:
            return None
        return self.completion_tokens / self.generation_seconds


@dataclass
class BenchmarkReport:
    model: str
    fast_path_ms: float
    idle_unloaded: IdleSample
    load: TimedStep
    idle_loaded: IdleSample
    requests: list[RequestSample] = field(default_factory=list)
    unload: TimedStep | None = None
    reload: TimedStep | None = None
    # First request after a reload: the real latency cost of having unloaded,
    # since llama.cpp's prompt cache is lost with the model.
    after_reload: RequestSample | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for sample, raw in zip(self.requests, data["requests"], strict=True):
            raw["generation_tokens_per_second"] = _round(sample.generation_tokens_per_second)
        return data

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def combined_stats(platform: PlatformAdapter, pids: Sequence[int | None]) -> ProcessStats | None:
    """Sum of several processes' stats (``None`` = this process), e.g. Jarvis + model worker."""
    samples = [s for s in (platform.process_stats(pid) for pid in pids) if s is not None]
    if not samples:
        return None
    return ProcessStats(
        rss_bytes=sum(s.rss_bytes for s in samples),
        footprint_bytes=sum(s.footprint_bytes or 0 for s in samples),
        wakeups=sum(s.wakeups or 0 for s in samples),
        cpu_seconds=sum(s.cpu_seconds or 0.0 for s in samples),
    )


def _idle(
    seconds: float,
    stats: Callable[[], ProcessStats | None],
    cpu_clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> IdleSample:
    before = Snapshot.take(stats)
    cpu0 = cpu_clock()
    sleep(seconds)
    cpu = cpu_clock() - cpu0
    after = Snapshot.take(stats)
    wakeups = None
    if before.wakeups is not None and after.wakeups is not None and seconds > 0:
        wakeups = round((after.wakeups - before.wakeups) / seconds, 2)
    return IdleSample(seconds, round(100 * cpu / seconds, 3) if seconds else 0.0, wakeups, after)


def _timed(
    action: Callable[[], object],
    stats: Callable[[], ProcessStats | None],
    clock: Callable[[], float],
    cpu_clock: Callable[[], float],
) -> TimedStep:
    t0, c0 = clock(), cpu_clock()
    action()
    cpu = max(0.0, cpu_clock() - c0)  # unloading removes the worker's CPU from the total
    return TimedStep(round(clock() - t0, 3), round(cpu, 3), Snapshot.take(stats))


def run_benchmark(
    models: ModelManager,
    planner: Planner,
    orchestrator: Orchestrator,
    stats: Callable[[], ProcessStats | None],
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    repeats: int = 2,
    idle_seconds: float = 10.0,
    progress: Callable[[str], None] = lambda _msg: None,
    clock: Callable[[], float] = time.monotonic,
    cpu_clock: Callable[[], float] = time.process_time,
    sleep: Callable[[float], None] = time.sleep,
) -> BenchmarkReport:
    """Run the benchmark. ``orchestrator`` must be built in dry-run mode."""
    process_clock = cpu_clock

    def combined_cpu() -> float:
        # Prefer the OS's CPU time for Jarvis *and* its model worker process;
        # time.process_time() only covers this process.
        sample = stats()
        if sample is not None and sample.cpu_seconds is not None:
            return sample.cpu_seconds
        return process_clock()

    cpu_clock = combined_cpu

    models.unload()

    progress("rule-based fast path")
    t0 = clock()
    orchestrator.handle("open Safari")
    fast_path_ms = round((clock() - t0) * 1000, 2)

    progress(f"idle {idle_seconds:g}s with model unloaded")
    idle_unloaded = _idle(idle_seconds, stats, cpu_clock, sleep)

    def load() -> None:
        with models.session():
            pass

    progress("loading model")
    load_step = _timed(load, stats, clock, cpu_clock)

    progress(f"idle {idle_seconds:g}s with model loaded")
    idle_loaded = _idle(idle_seconds, stats, cpu_clock, sleep)

    report = BenchmarkReport(
        model=models.backend.describe(),
        fast_path_ms=fast_path_ms,
        idle_unloaded=idle_unloaded,
        load=load_step,
        idle_loaded=idle_loaded,
    )

    def measure(prompt: str) -> RequestSample:
        t, c = clock(), cpu_clock()
        plan = planner.plan(prompt)
        wall, cpu = clock() - t, cpu_clock() - c
        last = plan.completions[-1]
        return RequestSample(
            prompt=prompt,
            wall_seconds=round(wall, 3),
            cpu_seconds=round(cpu, 3),
            action=plan.action.value,
            tools=[s.tool for s in plan.steps],
            attempts=plan.attempts,
            prompt_tokens=sum(x.prompt_tokens for x in plan.completions),
            cached_prompt_tokens=last.cached_prompt_tokens,
            completion_tokens=sum(x.completion_tokens for x in plan.completions),
            prompt_eval_seconds=_round(last.prompt_eval_seconds, 3),
            generation_seconds=_round(last.generation_seconds, 3),
        )

    for round_no in range(1, repeats + 1):
        for prompt in prompts:
            progress(f"round {round_no}: {prompt!r}")
            report.requests.append(measure(prompt))

    progress("unloading model")
    report.unload = _timed(models.unload, stats, clock, cpu_clock)
    progress("reloading model (file now in OS cache)")
    report.reload = _timed(load, stats, clock, cpu_clock)
    if prompts:
        progress("first request after reload")
        report.after_reload = measure(prompts[0])
    models.unload()
    return report


def render(report: BenchmarkReport) -> str:
    """Plain-text summary for the terminal."""

    def mem(s: Snapshot) -> str:
        if s.rss_mb is None:
            return "n/a"
        return f"RSS {s.rss_mb:,.0f} MB, footprint {s.footprint_mb or 0:,.0f} MB"

    def idle(label: str, s: IdleSample) -> str:
        wake = f", {s.wakeups_per_second:g} wakeups/s" if s.wakeups_per_second is not None else ""
        return f"  {label:<28} CPU {s.cpu_percent:.2f}%{wake}; {mem(s.memory)}"

    lines = [
        f"Model: {report.model}",
        "",
        "Idle (waiting for input):",
        idle("model unloaded", report.idle_unloaded),
        idle("model loaded", report.idle_loaded),
        "",
        "Model lifecycle:",
        f"  {'load (first in process)':<28} {report.load.wall_seconds:.2f}s wall, "
        f"{report.load.cpu_seconds:.2f}s CPU; {mem(report.load.memory_after)}",
    ]
    if report.unload:
        lines.append(
            f"  {'unload':<28} {report.unload.wall_seconds:.2f}s; {mem(report.unload.memory_after)}"
        )
    if report.reload:
        lines.append(
            f"  {'reload (warm file cache)':<28} {report.reload.wall_seconds:.2f}s wall, "
            f"{report.reload.cpu_seconds:.2f}s CPU"
        )
    if report.after_reload:
        lines.append(
            f"  {'first request after reload':<28} {report.after_reload.wall_seconds:.2f}s "
            "(prompt cache rebuilt)"
        )
    lines += ["", f"Rule-based fast path (no model): {report.fast_path_ms:.1f} ms", ""]
    lines.append("Planning requests:")
    for r in report.requests:
        tps = r.generation_tokens_per_second
        split = ""
        if r.prompt_eval_seconds is not None and r.generation_seconds is not None:
            split = f" (prompt {r.prompt_eval_seconds:.2f}s + gen {r.generation_seconds:.2f}s)"
        cached = f", {r.cached_prompt_tokens} cached" if r.cached_prompt_tokens is not None else ""
        lines.append(
            f"  {r.wall_seconds:5.2f}s{split}  {r.prompt_tokens} prompt tok{cached}, "
            f"{r.completion_tokens} gen tok"
            + (f" @ {tps:.0f} tok/s" if tps else "")
            + f"  -> {r.action} {', '.join(r.tools)}  [{r.prompt}]"
        )
    lines += [
        "",
        "Notes: CPU time excludes GPU (Metal) work; battery drain is not measured",
        "directly. Use `sudo powermetrics --samplers cpu_power,gpu_power` for power.",
    ]
    return "\n".join(lines)
