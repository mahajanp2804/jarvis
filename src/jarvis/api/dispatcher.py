"""Single choke point through which every tool call is executed.

Pipeline for each :class:`~jarvis.api.schemas.ToolCall`:

    look up tool -> validate arguments -> tool enabled? -> prepare (read-only:
    resolve targets, resource policy, per-call risk) -> decide (risk + taint)
    -> [dry run stops here] -> confirm if required -> run + verify
    (bounded retries for idempotent tools) -> audit

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
from dataclasses import dataclass

from pydantic import ValidationError

from jarvis.api.schemas import RiskLevel, ToolCall, ToolResult, ToolStatus
from jarvis.errors import (
    AmbiguousAppError,
    JarvisError,
    PolicyDeniedError,
    UnknownToolError,
)
from jarvis.runtime.audit import AuditLog
from jarvis.runtime.permissions import Confirmer, Decision, PolicyEngine
from jarvis.tools.base import PreparedAction, ToolRegistry, ToolSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionContext:
    """Per-request facts that affect policy.

    ``tainted``: the request has already read untrusted content (web pages,
    file contents, command output). Every later action that changes something
    then needs confirmation. See :meth:`PolicyEngine.decide`.
    """

    tainted: bool = False


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    backoff_seconds: float = 0.5


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
        retry: RetryPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.confirmer = confirmer
        self.audit = audit
        self.dry_run = dry_run
        self.retry = retry or RetryPolicy()
        # Set by the lifecycle manager (or Ctrl+C) to stop pending work safely.
        self.cancel_event = cancel_event or threading.Event()

    def execute(self, call: ToolCall, context: ExecutionContext | None = None) -> ToolResult:
        """Process one tool call end to end. Never raises for expected failures.

        ``KeyboardInterrupt`` is audited as a cancellation, sets the cancel
        event, and is re-raised so the caller can stop.
        """
        context = context or ExecutionContext()
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        trace: dict[str, str] = {"decision": "n/a", "risk": ""}
        result: ToolResult

        try:
            result = self._execute(call, context, trace)
        except JarvisError as exc:
            result = self._error_result(call, exc, trace)
        except KeyboardInterrupt:
            self.cancel_event.set()
            interrupted = ToolResult(
                tool=call.tool, status=ToolStatus.CANCELLED, message="Interrupted by the user."
            )
            self._record(request_id, call, trace, interrupted, started, context)
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

        self._record(request_id, call, trace, result, started, context)
        return result

    # -- pipeline ----------------------------------------------------------------------

    def _execute(
        self, call: ToolCall, context: ExecutionContext, trace: dict[str, str]
    ) -> ToolResult:
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

        if not self.policy.is_enabled(spec.name):
            trace["decision"] = Decision.DENY.value
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.DENIED,
                message=f"Not allowed: tool '{spec.name}' is not in policy.enabled_tools.",
                error_type="PolicyDeniedError",
            )

        prepared = spec.prepare(args)  # read-only; may raise JarvisError (incl. denial)
        risk = prepared.risk if prepared.risk is not None else spec.risk
        trace["risk"] = risk.name.lower()
        verdict = self.policy.decide(risk, tainted=context.tainted)
        trace["decision"] = verdict.decision.value
        needs_confirm = verdict.decision is Decision.CONFIRM and not prepared.noop

        if self.dry_run:
            note = " (would ask for confirmation)" if needs_confirm else ""
            return ToolResult(
                tool=call.tool,
                status=ToolStatus.DRY_RUN,
                message=f"[dry run] Would: {prepared.description}{note}",
            )

        if needs_confirm:
            prompt = f"Jarvis wants to: {prepared.description}"
            if prepared.warning:
                prompt += f"\n  Note: {prepared.warning}"
            if context.tainted and risk < RiskLevel.MEDIUM:
                prompt += "\n  (Asked because this request read web, file or command content.)"
            if not self.confirmer.confirm(prompt):
                trace["decision"] = "declined"
                return ToolResult(
                    tool=call.tool,
                    status=ToolStatus.CANCELLED,
                    message=f"Cancelled: {prepared.description} was not confirmed.",
                )
            trace["decision"] = "confirmed"

        return self._run_with_retries(spec, prepared)

    def _run_with_retries(self, spec: ToolSpec, prepared: PreparedAction) -> ToolResult:
        """Run once; retry transient failures of idempotent tools (bounded, with backoff)."""
        attempts_allowed = 1 + (self.retry.max_retries if spec.idempotent else 0)
        attempt = 0
        while True:
            attempt += 1
            if self.cancel_event.is_set():
                return ToolResult(
                    tool=spec.name,
                    status=ToolStatus.CANCELLED,
                    message="Cancelled: Jarvis is stopping.",
                    attempts=attempt,
                )
            try:
                result = prepared.run(self.cancel_event)
                transient = result.status is ToolStatus.FAILED and result.retryable
            except JarvisError as exc:
                if not exc.retryable or attempt >= attempts_allowed:
                    raise
                result = self._error_result_for(spec.name, exc)
                transient = True
            if not transient or attempt >= attempts_allowed:
                return result.model_copy(update={"attempts": attempt})
            delay = self.retry.backoff_seconds * attempt
            log.info("Retrying %s after transient failure (attempt %d)", spec.name, attempt)
            if self.cancel_event.wait(delay):
                return result.model_copy(update={"attempts": attempt})

    # -- results and audit ---------------------------------------------------------------

    @staticmethod
    def _error_result_for(tool: str, exc: JarvisError) -> ToolResult:
        return ToolResult(
            tool=tool,
            status=_status_for(exc),
            message=exc.message,
            hint=exc.hint,
            error_type=type(exc).__name__,
            retryable=exc.retryable,
        )

    def _error_result(self, call: ToolCall, exc: JarvisError, trace: dict[str, str]) -> ToolResult:
        result = self._error_result_for(call.tool, exc)
        if result.status is ToolStatus.DENIED:
            trace["decision"] = Decision.DENY.value
        return result

    def _record(
        self,
        request_id: str,
        call: ToolCall,
        trace: dict[str, str],
        result: ToolResult,
        started: float,
        context: ExecutionContext,
    ) -> None:
        duration_ms = (time.monotonic() - started) * 1000
        self.audit.record(
            request_id=request_id,
            tool=call.tool,
            arguments=dict(call.arguments),
            decision=trace["decision"],
            status=result.status.value,
            message=result.message,
            dry_run=self.dry_run,
            duration_ms=duration_ms,
            verified=result.verified,
            risk=trace["risk"] or None,
            attempts=result.attempts,
            tainted=context.tainted,
        )
        log.info(
            "tool=%s status=%s decision=%s risk=%s attempts=%d verified=%s %.0fms",
            call.tool,
            result.status.value,
            trace["decision"],
            trace["risk"] or "-",
            result.attempts,
            result.verified,
            duration_ms,
        )


def _status_for(exc: JarvisError) -> ToolStatus:
    if isinstance(exc, AmbiguousAppError):
        return ToolStatus.NEEDS_CLARIFICATION
    if isinstance(exc, PolicyDeniedError):
        return ToolStatus.DENIED
    return ToolStatus.FAILED
