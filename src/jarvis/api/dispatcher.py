"""Single choke point through which every tool call is executed.

Pipeline for each :class:`~jarvis.api.schemas.ToolCall`:

    look up tool -> validate arguments -> tool policy -> prepare (read-only,
    resource policy) -> [dry run stops here] -> confirm if required ->
    run + verify -> audit

Expected failures (:class:`~jarvis.errors.JarvisError`) become structured
results with hints. Unexpected exceptions are logged with a traceback and
reported generically, so a bug never surfaces as a raw stack trace to the user
and is never mistaken for success.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid

from pydantic import ValidationError

from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.errors import (
    AmbiguousAppError,
    JarvisError,
    PolicyDeniedError,
    UnknownToolError,
)
from jarvis.runtime.audit import AuditLog
from jarvis.runtime.permissions import Confirmer, Decision, PolicyEngine
from jarvis.tools.base import ToolRegistry

log = logging.getLogger(__name__)


def _format_arg_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{field}: {err['msg']}")
    return "; ".join(parts)


class Dispatcher:
    """Validates, authorises, confirms, executes and audits tool calls."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine,
        confirmer: Confirmer,
        audit: AuditLog,
        *,
        dry_run: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.confirmer = confirmer
        self.audit = audit
        self.dry_run = dry_run
        # Set by the lifecycle manager (or Ctrl+C) to stop pending work safely.
        self.cancel_event = cancel_event or threading.Event()

    def execute(self, call: ToolCall) -> ToolResult:
        """Process one tool call end to end. Never raises for expected failures.

        ``KeyboardInterrupt`` is audited as a cancellation, sets the cancel
        event, and is re-raised so the caller can stop.
        """
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        trace: dict[str, str] = {"decision": "n/a"}
        result: ToolResult

        try:
            result = self._execute(call, trace)
        except JarvisError as exc:
            status = _status_for(exc)
            if status is ToolStatus.DENIED:
                trace["decision"] = Decision.DENY.value
            result = ToolResult(
                tool=call.tool,
                status=status,
                message=exc.message,
                hint=exc.hint,
                error_type=type(exc).__name__,
            )
        except KeyboardInterrupt:
            self.cancel_event.set()
            interrupted = ToolResult(
                tool=call.tool, status=ToolStatus.CANCELLED, message="Interrupted by the user."
            )
            self._record(request_id, call, trace["decision"], interrupted, started)
            raise
        except Exception:
            log.exception("Unexpected error while executing %s", call.tool)
            result = ToolResult(
                tool=call.tool,
                status=ToolStatus.FAILED,
                message=f"Internal error while running '{call.tool}'.",
                hint="Details were written to the Jarvis log.",
                error_type="InternalError",
            )

        self._record(request_id, call, trace["decision"], result, started)
        return result

    def _record(
        self, request_id: str, call: ToolCall, decision: str, result: ToolResult, started: float
    ) -> None:
        duration_ms = (time.monotonic() - started) * 1000
        self.audit.record(
            request_id=request_id,
            tool=call.tool,
            arguments=dict(call.arguments),
            decision=decision,
            status=result.status.value,
            message=result.message,
            dry_run=self.dry_run,
            duration_ms=duration_ms,
            verified=result.verified,
        )
        log.info(
            "tool=%s status=%s decision=%s verified=%s %.0fms",
            call.tool,
            result.status.value,
            decision,
            result.verified,
            duration_ms,
        )

    def _execute(self, call: ToolCall, trace: dict[str, str]) -> ToolResult:
        try:
            spec = self.registry.get(call.tool)
        except UnknownToolError as exc:
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.FAILED,
                message=exc.message,
                hint="Available tools: " + ", ".join(self.registry.names()),
                error_type=type(exc).__name__,
            )

        try:
            args = spec.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.FAILED,
                message=f"Invalid arguments for '{call.tool}': {_format_arg_errors(exc)}",
                error_type="ToolInputError",
            )

        verdict = self.policy.evaluate_tool(spec.name, spec.risk)
        trace["decision"] = verdict.decision.value
        if verdict.decision is Decision.DENY:
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.DENIED,
                message=f"Not allowed: {verdict.reason}.",
                error_type="PolicyDeniedError",
            )

        prepared = spec.prepare(args)  # read-only; may raise JarvisError

        if self.dry_run:
            needs_confirm = verdict.decision is Decision.CONFIRM and not prepared.noop
            note = " (would ask for confirmation)" if needs_confirm else ""
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.DRY_RUN,
                message=f"[dry run] Would: {prepared.description}{note}",
            )

        if verdict.decision is Decision.CONFIRM and not prepared.noop:
            prompt = f"Jarvis wants to: {prepared.description}"
            if prepared.warning:
                prompt += f"\n  Note: {prepared.warning}"
            if not self.confirmer.confirm(prompt):
                trace["decision"] = "declined"
                return ToolResult(
                    tool=call.tool,
                    status=ToolStatus.CANCELLED,
                    message=f"Cancelled: {prepared.description} was not confirmed.",
                )
            trace["decision"] = "confirmed"

        if self.cancel_event.is_set():
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.CANCELLED,
                message="Cancelled: Jarvis is stopping.",
            )

        return prepared.run(self.cancel_event)


def _status_for(exc: JarvisError) -> ToolStatus:
    if isinstance(exc, AmbiguousAppError):
        return ToolStatus.NEEDS_CLARIFICATION
    if isinstance(exc, PolicyDeniedError):
        return ToolStatus.DENIED
    return ToolStatus.FAILED
