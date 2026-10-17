"""Per-call risk, taint escalation, tool wildcards and bounded retries."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from jarvis.api.dispatcher import Dispatcher, ExecutionContext, RetryPolicy
from jarvis.api.schemas import RiskLevel, ToolCall, ToolResult, ToolStatus
from jarvis.config import PolicyConfig
from jarvis.errors import ToolExecutionError
from jarvis.runtime.audit import AuditLog
from jarvis.runtime.permissions import Decision, PolicyEngine
from jarvis.tools.base import PreparedAction, ToolRegistry, ToolSpec
from tests.fakes import ScriptedConfirmer


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def make_tool(
    name: str,
    *,
    risk: RiskLevel = RiskLevel.LOW,
    call_risk: RiskLevel | None = None,
    idempotent: bool = False,
    outcomes: list[object] | None = None,
) -> tuple[ToolSpec, list[int]]:
    """A tool whose successive runs produce ``outcomes`` (ToolResult or exception)."""
    runs: list[int] = []
    queue = list(outcomes or [])

    def prepare(_args: NoArgs) -> PreparedAction:
        def run(_cancel: threading.Event) -> ToolResult:
            runs.append(1)
            outcome = queue.pop(0) if queue else None
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, ToolResult):
                return outcome
            return ToolResult(tool=name, status=ToolStatus.SUCCESS, message="done")

        return PreparedAction(description=f"Run {name}", run=run, risk=call_risk)

    return ToolSpec(name, "test tool", NoArgs, risk, prepare, idempotent=idempotent), runs


def dispatcher(
    tmp_path: Path, *specs: ToolSpec, confirm: bool = True, enabled: list[str] | None = None
) -> tuple[Dispatcher, ScriptedConfirmer]:
    confirmer = ScriptedConfirmer(confirm)
    d = Dispatcher(
        ToolRegistry(specs),
        PolicyEngine(PolicyConfig.model_validate({"enabled_tools": enabled or ["t.*"]})),
        confirmer,
        AuditLog(tmp_path / "audit.jsonl"),
        retry=RetryPolicy(max_retries=2, backoff_seconds=0.0),
    )
    return d, confirmer


def audit_lines(d: Dispatcher) -> list[dict[str, object]]:
    d.audit.close()
    return [json.loads(x) for x in d.audit.path.read_text(encoding="utf-8").splitlines()]


def call(name: str) -> ToolCall:
    return ToolCall(tool=name, arguments={})


# -- enablement --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("patterns", "tool", "enabled"),
    [
        (["apps.*"], "apps.open", True),
        (["apps.*"], "files.read", False),
        (["files.read"], "files.read", True),
        (["files.read"], "files.readme", False),
        (["*"], "anything.at.all", True),
        ([], "apps.open", False),
    ],
)
def test_wildcard_enablement(patterns: list[str], tool: str, enabled: bool) -> None:
    assert PolicyEngine(PolicyConfig(enabled_tools=patterns)).is_enabled(tool) is enabled


def test_disabled_tool_is_denied_before_prepare(tmp_path: Path) -> None:
    spec, runs = make_tool("x.run")
    d, _ = dispatcher(tmp_path, spec)
    assert d.execute(call("x.run")).status is ToolStatus.DENIED
    assert runs == []


# -- per-call risk -------------------------------------------------------------------------


def test_call_risk_can_raise_the_tool_default(tmp_path: Path) -> None:
    spec, runs = make_tool("t.cmd", risk=RiskLevel.LOW, call_risk=RiskLevel.HIGH)
    d, confirmer = dispatcher(tmp_path, spec, confirm=False)
    assert d.execute(call("t.cmd")).status is ToolStatus.CANCELLED
    assert len(confirmer.prompts) == 1 and runs == []
    assert audit_lines(d)[0]["risk"] == "high"


def test_call_risk_can_lower_the_tool_default(tmp_path: Path) -> None:
    spec, runs = make_tool("t.cmd", risk=RiskLevel.MEDIUM, call_risk=RiskLevel.READ_ONLY)
    d, confirmer = dispatcher(tmp_path, spec, confirm=False)
    assert d.execute(call("t.cmd")).status is ToolStatus.SUCCESS
    assert confirmer.prompts == [] and runs == [1]


# -- taint ---------------------------------------------------------------------------------


def test_tainted_request_needs_confirmation_for_any_change(tmp_path: Path) -> None:
    spec, runs = make_tool("t.open", risk=RiskLevel.LOW)
    d, confirmer = dispatcher(tmp_path, spec, confirm=False)
    clean = d.execute(call("t.open"))
    tainted = d.execute(call("t.open"), ExecutionContext(tainted=True))
    assert clean.status is ToolStatus.SUCCESS
    assert tainted.status is ToolStatus.CANCELLED
    assert "read web, file or command content" in confirmer.prompts[0]
    assert runs == [1]
    assert [e["tainted"] for e in audit_lines(d)] == [False, True]


def test_tainted_request_can_still_read(tmp_path: Path) -> None:
    spec, runs = make_tool("t.read", risk=RiskLevel.READ_ONLY)
    d, confirmer = dispatcher(tmp_path, spec, confirm=False)
    result = d.execute(call("t.read"), ExecutionContext(tainted=True))
    assert result.status is ToolStatus.SUCCESS and confirmer.prompts == []
    assert runs == [1]


def test_decide_matrix() -> None:
    policy = PolicyEngine(PolicyConfig())
    assert policy.decide(RiskLevel.LOW).decision is Decision.ALLOW
    assert policy.decide(RiskLevel.LOW, tainted=True).decision is Decision.CONFIRM
    assert policy.decide(RiskLevel.READ_ONLY, tainted=True).decision is Decision.ALLOW
    assert policy.decide(RiskLevel.MEDIUM).decision is Decision.CONFIRM


# -- retries -------------------------------------------------------------------------------


def transient() -> ToolExecutionError:
    return ToolExecutionError("timed out", retryable=True)


def test_idempotent_tool_retries_transient_errors(tmp_path: Path) -> None:
    spec, runs = make_tool("t.get", idempotent=True, outcomes=[transient(), transient()])
    d, _ = dispatcher(tmp_path, spec)
    result = d.execute(call("t.get"))
    assert result.status is ToolStatus.SUCCESS
    assert result.attempts == 3 and len(runs) == 3
    assert audit_lines(d)[0]["attempts"] == 3


def test_retries_are_bounded(tmp_path: Path) -> None:
    spec, runs = make_tool("t.get", idempotent=True, outcomes=[transient()] * 10)
    d, _ = dispatcher(tmp_path, spec)
    result = d.execute(call("t.get"))
    assert result.status is ToolStatus.FAILED
    assert result.retryable and len(runs) == 3


def test_returned_retryable_failures_are_retried(tmp_path: Path) -> None:
    flaky = ToolResult(tool="t.get", status=ToolStatus.FAILED, message="not yet", retryable=True)
    spec, runs = make_tool("t.get", idempotent=True, outcomes=[flaky])
    d, _ = dispatcher(tmp_path, spec)
    assert d.execute(call("t.get")).status is ToolStatus.SUCCESS
    assert len(runs) == 2


@pytest.mark.parametrize(
    ("idempotent", "error"),
    [
        (False, ToolExecutionError("timed out", retryable=True)),  # could apply twice
        (True, ToolExecutionError("no such file")),  # permanent
    ],
)
def test_no_retry_when_unsafe_or_permanent(
    tmp_path: Path, idempotent: bool, error: Exception
) -> None:
    spec, runs = make_tool("t.do", idempotent=idempotent, outcomes=[error])
    d, _ = dispatcher(tmp_path, spec)
    assert d.execute(call("t.do")).status is ToolStatus.FAILED
    assert len(runs) == 1


def test_cancellation_stops_retrying(tmp_path: Path) -> None:
    outcomes: list[object] = [transient()] * 5
    spec, runs = make_tool("t.get", idempotent=True, outcomes=outcomes)
    d, _ = dispatcher(tmp_path, spec)
    d.retry = RetryPolicy(max_retries=4, backoff_seconds=5.0)
    threading.Timer(0.1, d.cancel_event.set).start()
    result = d.execute(call("t.get"))
    assert len(runs) == 1  # the 5 s backoff was interrupted by the cancel event
    assert result.status is ToolStatus.FAILED
