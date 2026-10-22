from __future__ import annotations

from collections.abc import Callable

from jarvis.api.schemas import ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer, plan_json

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def test_rule_based_commands_skip_the_model(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("open Safari")
    assert reply.ok
    assert fake_llm.loads == 0
    assert fake_llm.calls == []


def test_multi_step_plan_executes_in_order(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.append(
        plan_json(
            "execute",
            "Opening.",
            ("apps.open", {"name": "Safari"}),
            ("apps.open", {"name": "Notes"}),
        )
    )
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("open my browser and notes")
    assert reply.ok
    assert reply.text == "Opened Safari.\nOpened Notes."
    assert fake_platform.apps.launched == ["Safari", "Notes"]


def test_failure_stops_the_round_and_the_model_sees_why(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.extend(
        [
            plan_json(
                "execute",
                "Working.",
                ("apps.open", {"name": "Safari"}),
                ("apps.open", {"name": "Nonexistent App"}),
                ("apps.open", {"name": "Notes"}),
            ),
            plan_json("respond", "I opened Safari, but there is no app called Nonexistent App."),
        ]
    )
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("open three things")
    assert not reply.ok
    assert len(reply.results) == 2  # Notes never ran: the round stopped at the failure
    assert fake_platform.apps.launched == ["Safari"]
    observation = fake_llm.calls[1][0][-1]["content"]
    assert "apps.open" in observation and "No installed application matches" in observation
    assert reply.text.startswith("I opened Safari, but there is no app")
    assert "✗ No installed application matches 'Nonexistent App'." in reply.text


def test_model_proposed_steps_still_need_confirmation(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("Notes")
    fake_llm.replies.append(plan_json("execute", "Quitting.", ("apps.quit", {"name": "Notes"})))
    runtime, confirmer = make_runtime(confirm=False)
    reply = runtime.orchestrator.handle("close my notes")
    assert "Cancelled" in reply.text
    assert len(confirmer.prompts) == 1
    assert fake_platform.apps.quit_requests == []


def test_model_cannot_bypass_policy(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.append(plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})))
    runtime, _ = make_runtime(policy={"app_blocklist": ["Safari"]})
    reply = runtime.orchestrator.handle("open the browser")
    assert "app_blocklist" in reply.text
    assert fake_platform.apps.launched == []


def test_dry_run_applies_to_planned_steps(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.append(plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})))
    runtime, _ = make_runtime(dry_run=True)
    reply = runtime.orchestrator.handle("open the browser")
    assert reply.text.startswith("[dry run] Would: Open Safari")
    assert fake_platform.apps.launched == []


def test_clarify_and_respond(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    fake_llm.replies.extend(
        [plan_json("clarify", "Which app?"), plan_json("respond", "I can't delete files yet.")]
    )
    runtime, _ = make_runtime()
    clarify = runtime.orchestrator.handle("open it")
    assert clarify.text == "Which app?"
    assert not clarify.handled
    respond = runtime.orchestrator.handle("delete my files")
    assert respond.text == "I can't delete files yet."
    assert respond.handled
    assert respond.results == []


def test_history_is_passed_to_the_model(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    fake_llm.replies.append(plan_json("execute", "Quitting.", ("apps.quit", {"name": "Notes"})))
    runtime, _ = make_runtime()
    runtime.orchestrator.handle("open notes")  # handled by rules, still remembered
    runtime.orchestrator.handle("now quit it")
    messages = fake_llm.calls[0][0]
    assert {"role": "user", "content": "open notes"} in messages
    assert {"role": "assistant", "content": "Opened Notes."} in messages
    runtime.orchestrator.clear_history()
    runtime.orchestrator.handle("hello")
    assert {"role": "user", "content": "open notes"} not in fake_llm.calls[1][0]


def test_planning_failure_is_reported(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    fake_llm.replies.extend(["garbage", "garbage"])
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("do something odd")
    assert not reply.handled
    assert "did not produce a valid plan" in reply.text


def test_ai_disabled(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    runtime, _ = make_runtime(ai={"enabled": False})
    reply = runtime.orchestrator.handle("open my browser")
    assert not reply.handled
    assert "disabled" in reply.text
    assert runtime.models is None
    assert fake_llm.loads == 0


def test_progress_hook(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    fake_llm.replies.append(plan_json("execute", "Opening.", ("apps.open", {"name": "Notes"})))
    runtime, _ = make_runtime()
    calls: list[str] = []
    runtime.orchestrator.on_progress = calls.append
    runtime.orchestrator.handle("open Safari")  # rules: no model, no progress
    assert calls == []
    runtime.orchestrator.handle("open my notes app please")
    assert calls == ["thinking", "apps.open (Notes)"]


def test_planner_sees_only_permitted_apps(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    runtime, _ = make_runtime(policy={"app_blocklist": ["Safari"]})
    runtime.orchestrator.handle("hello")
    system = fake_llm.calls[0][0][0]["content"]
    installed = next(x for x in system.splitlines() if x.startswith("Installed applications:"))
    assert "Notes" in installed
    assert "Safari" not in installed


def test_unknown_app_from_fast_path_falls_back_to_the_model(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.append(plan_json("execute", "Opening.", ("apps.open", {"name": "Safari"})))
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("open the browser")
    assert reply.text == "Opened Safari."
    assert fake_platform.apps.launched == ["Safari"]


def test_misheard_app_name_is_asked_about_without_the_model(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("Open photo wood.")  # as transcribed from "Photo Booth"
    assert reply.text == "There's no app called 'photo wood'. Did you mean Photo Booth?"
    assert reply.results[0].status is ToolStatus.NEEDS_CLARIFICATION
    assert fake_llm.calls == [] and fake_platform.apps.launched == []
    # The answer goes to the model with the question in the conversation.
    fake_llm.replies.append(
        plan_json("execute", "Opening.", ("apps.open", {"name": "Photo Booth"}))
    )
    assert runtime.orchestrator.handle("yes").text == "Opened Photo Booth."
    assert "Did you mean Photo Booth?" in str(fake_llm.calls[-1][0])


def test_unknown_app_without_ai_reports_not_found(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, _ = make_runtime(ai={"enabled": False})
    reply = runtime.orchestrator.handle("open the browser")
    assert "No installed application matches 'browser'" in reply.text


def test_ambiguous_fast_path_result_is_not_handed_to_the_model(
    make_runtime: MakeRuntime, fake_llm: FakeLLM
) -> None:
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("open photo")
    assert "Which one did you mean?" in reply.text
    assert fake_llm.calls == []


def test_history_expires_after_inactivity(make_runtime: MakeRuntime, fake_llm: FakeLLM) -> None:
    from jarvis.agent.orchestrator import Orchestrator

    runtime, _ = make_runtime()
    now = [0.0]
    orchestrator = Orchestrator(
        runtime.dispatcher,
        planner=runtime.planner,
        history_ttl_seconds=600,
        clock=lambda: now[0],
    )
    orchestrator.handle("open notes")
    now[0] = 100.0
    orchestrator.handle("hello")  # within the TTL: history included
    assert {"role": "user", "content": "open notes"} in fake_llm.calls[0][0]
    now[0] = 1000.0
    orchestrator.handle("now quit it")  # 15 minutes later: forgotten
    assert {"role": "user", "content": "open notes"} not in fake_llm.calls[1][0]
