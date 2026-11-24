"""`jarvis eval`: how well the planner picks actions and tools, on a fixed test set.

Each case in ``evals/planner_cases.toml`` is planned once (the first round
only) by the real model. Nothing is executed: the plan is only compared with
what the case expects (action, tools, key arguments). The installed-app list
comes from the test set, not this Mac, so scores are comparable between
machines and over time.

Run it before and after changing tools, prompts or models: a drop in a
category shows what a change made worse. Real-model tests check the overall
score against a floor.
"""

from __future__ import annotations

import json
import statistics
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from jarvis.agent.planner import Plan, Planner
from jarvis.errors import JarvisError

ACTIONS = ("execute", "clarify", "respond")


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    request: str
    actions: tuple[str, ...]
    tools: tuple[tuple[str, ...], ...] = ()  # acceptable first-round tool lists, sorted
    args: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    request: str
    passed: bool
    action: str | None
    tools: list[str]
    seconds: float
    problem: str | None = None
    profile: str = "all"  # tool profile the model saw


def _as_tuple(value: Any) -> tuple[str, ...]:
    return tuple(value) if isinstance(value, list) else (str(value),)


def load_cases(path: Path | None = None) -> tuple[list[str], list[Case]]:
    """The test set: (installed app names to pretend, cases)."""
    if path is None:
        text = resources.files("jarvis.evals").joinpath("planner_cases.toml").read_text("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    cases = []
    for raw in data.get("case", []):
        actions = _as_tuple(raw["action"])
        if any(a not in ACTIONS for a in actions):
            raise ValueError(f"case {raw['id']}: unknown action in {actions}")
        cases.append(
            Case(
                id=raw["id"],
                category=raw.get("category", "other"),
                request=raw["request"],
                actions=actions,
                tools=tuple(tuple(sorted(t)) for t in raw.get("tools", [])),
                args={
                    tool: {arg: _as_tuple(v) for arg, v in wanted.items()}
                    for tool, wanted in raw.get("args", {}).items()
                },
            )
        )
    ids = [c.id for c in cases]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates}")
    return list(data.get("apps", [])), cases


def score(case: Case, plan: Plan) -> str | None:
    """Why ``plan`` doesn't meet ``case`` (None if it does)."""
    action = plan.action.value
    if action not in case.actions:
        return f"action {action} ({plan.message!r}); expected {' or '.join(case.actions)}"
    tools = tuple(sorted(s.tool for s in plan.steps))
    if action != "execute":
        return None
    if case.tools and tools not in case.tools:
        expected = " or ".join("+".join(t) for t in case.tools)
        return f"tools {'+'.join(tools) or 'none'}; expected {expected}"
    for tool, wanted in case.args.items():
        for arg, needles in wanted.items():
            got = [str(s.arguments.get(arg, "")) for s in plan.steps if s.tool == tool]
            for needle in needles:
                options = [o.casefold() for o in needle.split("|")]
                if not any(o in g.casefold() for g in got for o in options):
                    return f"{tool} {arg}={got!r}; expected it to contain {needle!r}"
    return None


def run_eval(
    planner: Planner,
    cases: Sequence[Case],
    *,
    progress: Callable[[str], None] = lambda _msg: None,
    clock: Callable[[], float] = time.monotonic,
) -> list[CaseResult]:
    results = []
    for index, case in enumerate(cases, start=1):
        progress(f"{index}/{len(cases)} {case.id}")
        started = clock()
        try:
            plan = planner.plan(case.request)
        except JarvisError as exc:
            results.append(
                CaseResult(
                    case.id, case.category, case.request, False, None, [],
                    round(clock() - started, 2), f"no valid plan: {exc.message}",
                )
            )  # fmt: skip
            continue
        problem = score(case, plan)
        results.append(
            CaseResult(
                id=case.id,
                category=case.category,
                request=case.request,
                passed=problem is None,
                action=plan.action.value,
                tools=[s.tool for s in plan.steps],
                seconds=round(clock() - started, 2),
                problem=problem,
                profile=plan.profile,
            )
        )
    return results


def pass_rate(results: Sequence[CaseResult]) -> float:
    return sum(r.passed for r in results) / len(results) if results else 0.0


def render(results: Sequence[CaseResult]) -> str:
    if not results:
        return "No cases."
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    lines = [f"{'category':<10} {'passed':>9}"]
    for category in sorted(by_category):
        group = by_category[category]
        lines.append(f"{category:<10} {sum(r.passed for r in group):>4}/{len(group):<4}")
    seconds = sorted(r.seconds for r in results)
    later = seconds[1:] or seconds  # the first case also reads the system prompt
    lines += [
        f"{'overall':<10} {sum(r.passed for r in results):>4}/{len(results):<4}"
        f" ({pass_rate(results):.0%})",
        "",
        f"Planning time: first {results[0].seconds:.1f}s (reads the system prompt), "
        f"then median {statistics.median(later):.1f}s, slowest {max(later):.1f}s",
    ]
    profiles = Counter(r.profile for r in results)
    if set(profiles) != {"all"}:
        shown = ", ".join(f"{name} {count}" for name, count in sorted(profiles.items()))
        lines.append(f"Tool profiles: {shown}")
    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "Failed:"]
        lines += [f"  {r.id:<26} {r.request!r}\n  {'':<26} {r.problem}" for r in failed]
    return "\n".join(lines)


def write_json(results: Sequence[CaseResult], path: Path) -> None:
    data = {"pass_rate": round(pass_rate(results), 4), "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
