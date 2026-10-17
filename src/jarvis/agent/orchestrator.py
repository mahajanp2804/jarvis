"""Agent orchestrator: turns a user request into tool calls and a reply.

1. The rule-based router handles simple commands instantly, without loading the
   model. If the router's guess names an app that doesn't exist ("open the
   browser"), nothing has happened yet, so the request falls through to step 2.
2. Anything else goes to the LLM planner (if AI is enabled), which proposes a
   validated plan: execute steps, ask a clarifying question, or just respond.
3. Steps run one at a time through the dispatcher (validation, policy,
   confirmation, verification, audit). Execution stops at the first step that
   does not succeed, and the reply says what was skipped.

A short in-memory conversation history lets follow-ups like "now quit it" work.
It is never written to disk (persistent memory arrives in Phase 5).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from jarvis.agent.planner import Plan, PlanAction, Planner
from jarvis.agent.router import route
from jarvis.ai.base import ChatMessage
from jarvis.api.dispatcher import Dispatcher
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.errors import JarvisError

HELP_TEXT = """\
Instant commands (no AI needed):
  open <app>              e.g. "open Safari", "launch the Notes app"
  quit <app>              e.g. "quit Music" (asks for confirmation)
  list apps [matching X]  e.g. "list apps matching studio"
  find app <name>         e.g. "find app code"
Anything else is planned by the local AI model, e.g. "open my browser and notes"."""

# Fast-path failures that changed nothing and that the model may resolve better
# (e.g. "browser" is not an app name, but the model can map it to Safari).
_FALL_THROUGH_ERRORS = {"AppNotFoundError"}

AI_DISABLED_TEXT = (
    "I can't handle that with the built-in rules, and the AI model is disabled "
    "([ai] enabled = false). Type 'help' to see what works without it."
)


@dataclass
class Reply:
    """Outcome of handling one user request."""

    text: str
    results: list[ToolResult] = field(default_factory=list)
    handled: bool = True  # False: nothing was attempted (not understood / clarification)
    plan: Plan | None = None

    @property
    def ok(self) -> bool:
        return self.handled and all(r.ok for r in self.results)


class Orchestrator:
    """Routes requests to tools via the rule router, then the LLM planner."""

    def __init__(
        self,
        dispatcher: Dispatcher,
        router: Callable[[str], ToolCall | None] = route,
        planner: Planner | None = None,
        *,
        history_turns: int = 3,
        history_ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dispatcher = dispatcher
        self._router = router
        self._planner = planner
        # Forget the conversation after this much inactivity, so an old exchange
        # can't change the meaning of a new request ("quit it" an hour later).
        self._history_ttl = history_ttl_seconds
        self._clock = clock
        self._last_activity: float | None = None
        # Called just before the (slower) model is consulted, e.g. to show "thinking".
        self.on_planning: Callable[[], None] | None = None
        self._history: deque[ChatMessage] = deque(maxlen=history_turns * 2)

    def handle(self, text: str) -> Reply:
        """Handle one natural-language request."""
        text = text.strip()
        if not text:
            return Reply("Say something like 'open Safari' (or 'help').", handled=False)
        if text.lower() in {"help", "?", "what can you do"}:
            return Reply(HELP_TEXT)
        now = self._clock()
        if self._last_activity is not None and now - self._last_activity > self._history_ttl:
            self._history.clear()
        self._last_activity = now

        call = self._router(text)
        result = self._dispatcher.execute(call) if call is not None else None
        if result is not None and not (
            self._planner is not None and result.error_type in _FALL_THROUGH_ERRORS
        ):
            reply = Reply(format_result(result), [result])
        elif self._planner is None:
            reply = Reply(AI_DISABLED_TEXT, handled=False)
        else:
            reply = self._plan_and_execute(self._planner, text)

        self._remember(text, reply.text)
        return reply

    def clear_history(self) -> None:
        self._history.clear()

    def _plan_and_execute(self, planner: Planner, text: str) -> Reply:
        if self.on_planning is not None:
            self.on_planning()
        try:
            plan = planner.plan(text, list(self._history))
        except JarvisError as exc:
            message = exc.message + (f"\n  Hint: {exc.hint}" if exc.hint else "")
            return Reply(message, handled=False)

        if plan.action is PlanAction.CLARIFY:
            return Reply(plan.message, handled=False, plan=plan)
        if plan.action is PlanAction.RESPOND:
            return Reply(plan.message, plan=plan)

        results: list[ToolResult] = []
        for step in plan.steps:
            result = self._dispatcher.execute(step)
            results.append(result)
            if not result.ok:
                break
        lines = [format_result(r) for r in results]
        skipped = plan.steps[len(results) :]
        if skipped:
            names = ", ".join(_describe_call(c) for c in skipped)
            lines.append(f"Skipped {len(skipped)} remaining step(s): {names}.")
        return Reply("\n".join(lines), results, plan=plan)

    def _remember(self, request: str, reply: str) -> None:
        self._history.append({"role": "user", "content": request})
        self._history.append({"role": "assistant", "content": reply})


def _describe_call(call: ToolCall) -> str:
    target = call.arguments.get("name") or call.arguments.get("query")
    return f"{call.tool}({target})" if target else call.tool


def format_result(result: ToolResult) -> str:
    """Human-readable one-block summary of a tool result."""
    text = result.message
    if result.status is ToolStatus.NEEDS_CLARIFICATION:
        text = f"{text}\nWhich one did you mean?"
    if result.hint:
        text = f"{text}\n  Hint: {result.hint}"
    return text
