from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis import main as cli
from jarvis.agent.planner import Plan, PlanAction
from jarvis.api.schemas import ToolCall
from jarvis.bootstrap import Runtime
from jarvis.evaluation import Case, load_cases, render, run_eval, score
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer, plan_json

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def plan(action: str, *steps: tuple[str, dict[str, object]]) -> Plan:
    return Plan(
        PlanAction(action),
        "ok",
        tuple(ToolCall(tool=t, arguments=a) for t, a in steps),
        attempts=1,
        completions=(),
    )


def test_bundled_test_set_names_real_tools_and_arguments(
    make_runtime: MakeRuntime, tmp_path: Path
) -> None:
    from jarvis.config import BrowserConfig
    from jarvis.tools.browser import BrowserSession, BrowserTools

    apps, cases = load_cases()
    assert len(cases) >= 90 and "Safari" in apps
    runtime, _ = make_runtime(policy={"enabled_tools": ["*"]})
    specs = {spec.name: spec for spec in runtime.registry}
    # Browser tools are only registered when Playwright is installed (not on Linux CI);
    # their definitions don't need it.
    browser = BrowserTools(BrowserSession(BrowserConfig(), tmp_path / "profile"), BrowserConfig())
    specs.update({spec.name: spec for spec in browser.specs()})
    for case in cases:
        for tools in case.tools:
            assert set(tools) <= set(specs), case.id
        for tool, wanted in case.args.items():
            fields = specs[tool].args_model.model_fields
            assert set(wanted) <= set(fields), (
                f"{case.id}: {tool} has no {set(wanted) - set(fields)}"
            )
    categories = {c.category for c in cases}
    assert {"apps", "files", "projects", "browser", "siri", "chat", "vague", "safety"} <= categories


def test_scoring_rules() -> None:
    case = Case(
        "c", "apps", "open notes and calculator", ("execute",),
        tools=(("apps.open", "apps.open"),),
        args={"apps.open": {"name": ("notes", "calculator|calc")}},
    )  # fmt: skip
    good = plan("execute", ("apps.open", {"name": "Calculator"}), ("apps.open", {"name": "Notes"}))
    assert score(case, good) is None  # order doesn't matter
    one = plan("execute", ("apps.open", {"name": "Notes"}))
    assert "tools apps.open; expected apps.open+apps.open" in str(score(case, one))
    wrong = plan("execute", ("apps.open", {"name": "Notes"}), ("apps.open", {"name": "Safari"}))
    assert "expected it to contain 'calculator|calc'" in str(score(case, wrong))
    asked = plan("clarify")
    assert "action clarify" in str(score(case, asked))

    decline = Case("d", "siri", "set a timer", ("respond",))
    assert score(decline, plan("respond")) is None
    assert "expected respond" in str(
        score(decline, plan("execute", ("apps.open", {"name": "Clock"})))
    )


def test_run_eval_and_report(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    runtime, _ = make_runtime()
    assert runtime.planner is not None
    fake_llm.replies.extend(
        [
            plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})),
            plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})),  # wrong
            plan_json("respond", "I can't set timers yet."),
        ]
    )
    cases = [
        Case(
            "open", "apps", "open safari", ("execute",), (("apps.open",),),
            {"apps.open": {"name": ("safari",)}},
        ),
        Case("quit", "apps", "quit music", ("execute",), (("apps.quit",),)),
        Case("timer", "siri", "set a timer", ("respond",)),
    ]  # fmt: skip
    results = run_eval(runtime.planner, cases)
    assert [r.passed for r in results] == [True, False, True]
    assert fake_llm.calls and all(
        len(c[0]) == 2 for c in fake_llm.calls
    )  # no history between cases
    text = " ".join(render(results).split())
    assert "apps 1/2" in text and "siri 1/1" in text
    assert "overall 2/3 (67%)" in text and "quit music" in text


def test_eval_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLLM(*[plan_json("respond", "Hello.")] * 3)
    monkeypatch.setattr("jarvis.bootstrap.build_backend", lambda ai: fake)
    platform = FakePlatform()
    monkeypatch.setattr(cli, "get_platform", lambda: platform)
    monkeypatch.setattr("jarvis.config.default_config_file", lambda: tmp_path / "config.toml")
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("JARVIS_MODEL_PATH", str(model))
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "logs"))
    out = tmp_path / "eval.json"
    code = cli.main(["eval", "--category", "chat", "--limit", "3", "--json", str(out)])
    assert code == cli.EXIT_OK
    text = " ".join(capsys.readouterr().out.split())
    assert "Planning 3 test requests" in text and "chat 3/3" in text
    data = json.loads(out.read_text())
    assert data["pass_rate"] == 1.0 and len(data["results"]) == 3
    assert platform.apps.launched == []  # nothing was executed
    # A floor turns a weak score into exit code 3.
    fake.replies.extend([plan_json("execute", "x", ("apps.open", {"name": "Safari"}))] * 3)
    assert cli.main(["eval", "--category", "chat", "--limit", "3", "--min-pass", "0.9"]) == 3
