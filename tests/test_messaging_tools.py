"""messages.send and mail.* on the fake apps: nothing is ever sent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(runtime: Runtime, tool: str, **args: Any) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=args))


def test_sending_always_asks_and_shows_exactly_what_and_to_whom(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    declined = run(runtime, "messages.send", to="mom", text="I'll be late")
    assert declined.status is ToolStatus.CANCELLED and fake_platform.outbox.texts == []
    prompt = confirmer.prompts[0]
    assert "Send a message to Jane Doe (+1 555 0100): “I'll be late”" in prompt
    assert "can't be unsent" in prompt

    runtime, confirmer = make_runtime(confirm=True)
    sent = run(runtime, "messages.send", to="+1 555 0177", text="On my way")
    assert sent.message == "Sent your message to +1 555 0177 by iMessage."
    assert fake_platform.outbox.texts == [("+1 555 0177", "On my way")]
    assert len(confirmer.prompts) == 1  # asked again: approvals aren't remembered


def test_no_policy_can_skip_the_confirmation(make_runtime: MakeRuntime) -> None:
    with pytest.raises(ValidationError, match="always require confirmation"):
        make_runtime(policy={"confirm_at_or_above": "critical"})
    runtime, confirmer = make_runtime(confirm=False, policy={"confirm_at_or_above": "high"})
    result = run(runtime, "messages.send", to="mom", text="hi")
    assert result.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1


def test_unclear_recipients_ask_before_any_confirmation(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=True)
    result = run(runtime, "messages.send", to="Sam", text="hi")
    assert result.status is ToolStatus.NEEDS_CLARIFICATION
    assert result.message == "Which Sam: Sam Lee, Sam Patel?"
    assert confirmer.prompts == [] and fake_platform.outbox.texts == []


def test_email_draft_and_send(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    draft = run(runtime, "mail.draft", to="Sam Lee", subject="Sick today", body="Back tomorrow.")
    assert draft.message == "Opened an email to Sam Lee in Mail: check it and press Send."
    assert confirmer.prompts == []  # a draft sends nothing
    declined = run(runtime, "mail.send", to="Sam Lee", subject="Sick today", body="Back tomorrow.")
    assert declined.status is ToolStatus.CANCELLED
    assert (
        "Send an email to Sam Lee (sam@example.org): “Sick today”, saying “Back tomorrow.”"
        in (confirmer.prompts[0])
    )
    assert fake_platform.outbox.emails == [
        ("sam@example.org", "Sick today", "Back tomorrow.", False)
    ]
