from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import pytest

from jarvis.api.schemas import ToolCall, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.errors import ToolExecutionError
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def audit_entries(runtime: Runtime) -> list[dict[str, Any]]:
    runtime.audit.close()
    text = runtime.audit.path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_unknown_tool(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="files.delete", arguments={"path": "/"}))
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "UnknownToolError"
    assert result.hint and "apps.open" in result.hint
    assert audit_entries(runtime)[0]["status"] == "failed"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"name": ""},
        {"name": "   "},
        {"name": "Safari\x00"},
        {"name": "x" * 500},
        {"name": "Safari", "force": True},
        {"name": 42},
    ],
)
def test_invalid_arguments_are_rejected_before_anything_runs(
    make_runtime: MakeRuntime, fake_platform: FakePlatform, arguments: dict[str, Any]
) -> None:
    runtime, _ = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments=arguments))
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "ToolInputError"
    assert fake_platform.apps.launched == []
    assert fake_platform.apps.discover_calls == 0


def test_tool_call_name_is_validated() -> None:
    with pytest.raises(ValueError, match="tool"):
        ToolCall(tool="apps.open; rm -rf /")


def test_low_risk_runs_without_confirmation(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "safari"}))
    assert result.status is ToolStatus.SUCCESS
    assert result.verified is True
    assert confirmer.prompts == []
    assert fake_platform.apps.launched == ["Safari"]
    entry = audit_entries(runtime)[0]
    assert entry["decision"] == "allow"
    assert entry["verified"] is True


def test_dry_run_never_executes(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    fake_platform.apps.running.add("Notes")
    runtime, confirmer = make_runtime(dry_run=True)
    opened = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    quit_ = runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "Notes"}))
    assert opened.status is ToolStatus.DRY_RUN
    assert "Open Safari" in opened.message
    assert quit_.status is ToolStatus.DRY_RUN
    assert "would ask for confirmation" in quit_.message
    assert fake_platform.apps.launched == []
    assert fake_platform.apps.quit_requests == []
    assert confirmer.prompts == []
    assert all(e["dry_run"] for e in audit_entries(runtime))


def test_declined_confirmation_cancels(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("Notes")
    runtime, confirmer = make_runtime(confirm=False)
    result = runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "notes"}))
    assert result.status is ToolStatus.CANCELLED
    assert fake_platform.apps.quit_requests == []
    assert "Quit Notes" in confirmer.prompts[0]
    assert "save" in confirmer.prompts[0]
    assert audit_entries(runtime)[0]["decision"] == "declined"


def test_confirmed_action_runs_and_is_verified(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("Notes")
    runtime, confirmer = make_runtime(confirm=True)
    result = runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "Notes"}))
    assert result.status is ToolStatus.SUCCESS
    assert result.verified is True
    assert fake_platform.apps.quit_requests == ["Notes"]
    assert len(confirmer.prompts) == 1
    assert audit_entries(runtime)[0]["decision"] == "confirmed"


def test_noop_skips_confirmation(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "Notes"}))
    assert result.status is ToolStatus.SUCCESS
    assert "isn't running" in result.message
    assert confirmer.prompts == []
    assert fake_platform.apps.quit_requests == []


def test_each_sensitive_action_needs_its_own_confirmation(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=True)
    for _ in range(2):
        fake_platform.apps.running.add("Notes")
        runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "Notes"}))
    assert len(confirmer.prompts) == 2


def test_disabled_tool_is_denied(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, _ = make_runtime(policy={"enabled_tools": ["apps.list"]})
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    assert result.status is ToolStatus.DENIED
    assert fake_platform.apps.launched == []
    assert audit_entries(runtime)[0]["decision"] == "deny"


def test_blocklisted_app_is_denied(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, _ = make_runtime(policy={"app_blocklist": ["com.apple.Safari"]})
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    assert result.status is ToolStatus.DENIED
    assert "app_blocklist" in result.message
    assert fake_platform.apps.launched == []
    assert audit_entries(runtime)[0]["decision"] == "deny"


def test_expected_error_becomes_failed_result_with_hint(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.launch_error = ToolExecutionError("open failed", hint="try Finder")
    runtime, _ = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    assert result.status is ToolStatus.FAILED
    assert result.message == "open failed"
    assert result.hint == "try Finder"


def test_unexpected_error_is_contained(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.launch_error = RuntimeError("secret internal detail token=abcdef123456")
    runtime, _ = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    assert result.status is ToolStatus.FAILED
    assert result.error_type == "InternalError"
    assert "secret internal detail" not in result.message


def test_cancel_event_stops_pending_action(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    cancel = threading.Event()
    cancel.set()
    runtime, _ = make_runtime(cancel_event=cancel)
    result = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "Safari"}))
    assert result.status is ToolStatus.CANCELLED
    assert fake_platform.apps.launched == []


def test_keyboard_interrupt_is_audited_and_propagates(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("Notes")
    runtime, confirmer = make_runtime()

    def interrupt(prompt: str) -> bool:
        raise KeyboardInterrupt

    confirmer.confirm = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        runtime.dispatcher.execute(ToolCall(tool="apps.quit", arguments={"name": "Notes"}))
    assert runtime.dispatcher.cancel_event.is_set()
    assert fake_platform.apps.quit_requests == []
    assert audit_entries(runtime)[0]["status"] == "cancelled"
