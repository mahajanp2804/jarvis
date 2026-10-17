"""Permission and execution policy.

The policy layer is the single place that decides whether an action may run,
must be confirmed by the user, or is denied. It never trusts the origin of a
request: tool calls built by the rule-based router and (later) by the language
model are judged identically.

Approvals are never cached. Each confirmation covers exactly one action, so a
past "yes" can never authorise a future sensitive action.
"""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from jarvis.api.schemas import RiskLevel
from jarvis.config import PolicyConfig


class Decision(StrEnum):
    """Outcome of a policy evaluation."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyResult:
    """A policy decision with a human-readable reason (recorded in the audit log)."""

    decision: Decision
    reason: str


class PolicyEngine:
    """Evaluates tool calls against :class:`~jarvis.config.PolicyConfig`."""

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config
        self._allowlist = {item.casefold() for item in config.app_allowlist}
        self._blocklist = {item.casefold() for item in config.app_blocklist}

    def is_enabled(self, tool_name: str) -> bool:
        """Whether ``policy.enabled_tools`` lists the tool (``*`` wildcards allowed)."""
        patterns = self._config.enabled_tools
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in patterns)

    def decide(self, risk: RiskLevel, *, tainted: bool = False) -> PolicyResult:
        """Allow or require confirmation for an action of ``risk``.

        ``tainted`` means the current request has already read untrusted
        content (a web page, a file, command output). Then any action that
        changes something needs confirmation, so instructions hidden in that
        content can't trigger actions on their own.
        """
        # HIGH and CRITICAL always need confirmation, whatever the configured threshold.
        threshold = min(self._config.confirm_at_or_above, RiskLevel.HIGH)
        if risk >= threshold:
            return PolicyResult(
                Decision.CONFIRM, f"risk '{risk.name.lower()}' requires confirmation"
            )
        if tainted and risk > RiskLevel.READ_ONLY:
            return PolicyResult(
                Decision.CONFIRM, "the request has read untrusted content (web, files or output)"
            )
        return PolicyResult(Decision.ALLOW, f"risk '{risk.name.lower()}' is below threshold")

    def evaluate_tool(
        self, tool_name: str, risk: RiskLevel, *, tainted: bool = False
    ) -> PolicyResult:
        """Decide whether a tool at a given risk level may run."""
        if not self.is_enabled(tool_name):
            return PolicyResult(Decision.DENY, f"tool '{tool_name}' is not in policy.enabled_tools")
        return self.decide(risk, tainted=tainted)

    def check_app(self, name: str, bundle_id: str | None) -> PolicyResult:
        """Apply the application allow/block lists to a resolved application."""
        identifiers = {name.casefold()}
        if bundle_id:
            identifiers.add(bundle_id.casefold())
        if identifiers & self._blocklist:
            return PolicyResult(Decision.DENY, f"'{name}' is in policy.app_blocklist")
        if self._allowlist and not identifiers & self._allowlist:
            return PolicyResult(Decision.DENY, f"'{name}' is not in policy.app_allowlist")
        return PolicyResult(Decision.ALLOW, "application permitted")


class Confirmer(Protocol):
    """Asks the user to approve one specific action."""

    def confirm(self, prompt: str) -> bool:
        """Return True only if the user explicitly approved."""
        ...


class DenyAllConfirmer:
    """Used when nobody is available to answer (non-interactive runs)."""

    def confirm(self, prompt: str) -> bool:
        return False


class InteractiveConfirmer:
    """Prompts on the terminal. Anything other than an explicit yes is a no."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        is_interactive: Callable[[], bool] = lambda: sys.stdin.isatty(),
    ) -> None:
        self._input = input_fn
        self._is_interactive = is_interactive

    def confirm(self, prompt: str) -> bool:
        if not self._is_interactive():
            return False
        try:
            answer = self._input(f"{prompt}\nProceed? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}
