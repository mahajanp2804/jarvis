"""Activation flow with a scripted UI on the portable event loop."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import pytest

from jarvis.agent.orchestrator import Reply
from jarvis.api.schemas import ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.runtime.assistant import Assistant, DialogConfirmer, needs_answer, should_show
from jarvis.runtime.event_loop import QueueEventLoop
from jarvis.ui.base import UserInterface
from tests.fakes import FakeLLM, FakePlatform, ScriptedConfirmer, plan_json

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


class ScriptedUI(UserInterface):
    """Answers prompts from a queue; records everything; runs on the loop thread."""

    def __init__(self, *answers: str | None, confirm: bool = False) -> None:
        self.answers: deque[str | None] = deque(answers)
        self.confirm_answer = confirm
        self.prompts: list[str] = []
        self.confirms: list[str] = []
        self.shown: list[str] = []
        self.threads: set[bool] = set()

    def prompt(self, title: str, message: str) -> str | None:
        self.threads.add(threading.current_thread() is threading.main_thread())
        self.prompts.append(message)
        return self.answers.popleft() if self.answers else None

    def confirm(self, title: str, message: str) -> bool:
        self.threads.add(threading.current_thread() is threading.main_thread())
        self.confirms.append(message)
        return self.confirm_answer

    def show(self, title: str, message: str) -> None:
        self.threads.add(threading.current_thread() is threading.main_thread())
        self.shown.append(message)


def drive(
    assistant: Assistant, loop: QueueEventLoop, action: Callable[[], None] | None = None
) -> None:
    """Run the loop on the main thread until the assistant's request is done."""

    def start() -> None:
        (action or assistant.activate)()

        def wait_and_stop() -> None:
            assistant.shutdown(timeout=5)
            loop.stop()

        threading.Thread(target=wait_and_stop, daemon=True).start()

    loop.call_soon(start)
    loop.run()


def make(
    make_runtime: MakeRuntime, ui: ScriptedUI, **kwargs: object
) -> tuple[Assistant, QueueEventLoop, Runtime]:
    loop = QueueEventLoop()
    runtime, _ = make_runtime()
    runtime.dispatcher.confirmer = DialogConfirmer(loop, ui)
    return Assistant(runtime.orchestrator, ui, loop, **kwargs), loop, runtime  # type: ignore[arg-type]


def test_visible_success_shows_no_dialog(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    ui = ScriptedUI("open safari")
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert fake_platform.apps.launched == ["Safari"]
    assert ui.shown == []
    assert assistant.requests_handled == 1
    assert ui.threads == {True}  # every dialog ran on the main thread


def test_show_success_option(make_runtime: MakeRuntime) -> None:
    ui = ScriptedUI("open safari")
    assistant, loop, _ = make(make_runtime, ui, show_success=True)
    drive(assistant, loop)
    assert ui.shown == ["Opened Safari."]


def test_text_reply_is_shown(make_runtime: MakeRuntime) -> None:
    ui = ScriptedUI("list apps matching photo")
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert len(ui.shown) == 1 and "Photo Booth" in ui.shown[0]


def test_cancelled_prompt_does_nothing(make_runtime: MakeRuntime) -> None:
    ui = ScriptedUI(None)
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert assistant.requests_handled == 0
    assert ui.shown == []


def test_clarifying_question_is_asked_and_answered(
    make_runtime: MakeRuntime, fake_llm: FakeLLM, fake_platform: FakePlatform
) -> None:
    fake_llm.replies.extend(
        [
            plan_json("clarify", "Which app should I open?"),
            plan_json("execute", "Opening.", ("apps.open", {"name": "Notes"})),
        ]
    )
    ui = ScriptedUI("open it", "the notes one")
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert ui.prompts == ["What should I do?", "Which app should I open?"]
    assert fake_platform.apps.launched == ["Notes"]


def test_confirmation_dialog_runs_on_main_thread_and_defaults_to_no(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("Notes")
    ui = ScriptedUI("quit notes", confirm=False)
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert len(ui.confirms) == 1 and "Quit Notes" in ui.confirms[0]
    assert fake_platform.apps.quit_requests == []
    assert ui.shown == []  # declining needs no further dialog
    assert ui.threads == {True}


def test_confirmation_allowed(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    fake_platform.apps.running.add("Notes")
    ui = ScriptedUI("quit notes", confirm=True)
    assistant, loop, _ = make(make_runtime, ui)
    drive(assistant, loop)
    assert fake_platform.apps.quit_requests == ["Notes"]


def test_non_interactive_requests_never_open_dialogs(make_runtime: MakeRuntime) -> None:
    ui = ScriptedUI()
    assistant, loop, _ = make(make_runtime, ui)
    replies: list[Reply] = []

    def ask() -> None:
        future = assistant.submit("list apps", interactive=False)
        future.add_done_callback(lambda f: replies.append(f.result()))

    drive(assistant, loop, ask)
    assert "Safari" in replies[0].text
    assert ui.prompts == [] and ui.shown == []


def test_busy_assistant_says_so(make_runtime: MakeRuntime) -> None:
    gate = threading.Event()
    ui = ScriptedUI("anything")
    assistant, loop, runtime = make(make_runtime, ui)
    original = runtime.orchestrator.handle

    def slow(text: str, **kwargs: Any) -> Reply:
        gate.wait(5)
        return original(text, **kwargs)

    runtime.orchestrator.handle = slow  # type: ignore[method-assign]

    def start() -> None:
        assistant.activate()  # starts the slow request
        assistant.activate()  # second press while busy
        gate.set()

    drive(assistant, loop, start)
    assert "Still working" in ui.shown[0]


@pytest.mark.parametrize(
    ("results", "shown"),
    [
        ([], True),
        ([("apps.open", ToolStatus.SUCCESS, True)], False),
        ([("apps.open", ToolStatus.SUCCESS, None)], True),  # unverified
        ([("apps.list", ToolStatus.SUCCESS, None)], True),
        ([("apps.open", ToolStatus.FAILED, False)], True),
        ([("apps.quit", ToolStatus.CANCELLED, None)], False),
        ([("apps.open", ToolStatus.DRY_RUN, None)], True),
        ([("apps.open", ToolStatus.DENIED, None)], True),
    ],
)
def test_should_show(results: list[tuple[str, ToolStatus, bool | None]], shown: bool) -> None:
    reply = Reply(
        "text",
        [ToolResult(tool=t, status=s, message="m", verified=v) for t, s, v in results],
    )
    assert should_show(reply) is shown


def test_needs_answer() -> None:
    assert needs_answer(
        Reply(
            "?", [ToolResult(tool="apps.open", status=ToolStatus.NEEDS_CLARIFICATION, message="")]
        )
    )
    assert not needs_answer(Reply("ok", []))
