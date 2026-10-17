from __future__ import annotations

import pytest

from jarvis.api.schemas import RiskLevel
from jarvis.config import PolicyConfig
from jarvis.runtime.permissions import (
    Decision,
    DenyAllConfirmer,
    InteractiveConfirmer,
    PolicyEngine,
)


def engine(**kwargs: object) -> PolicyEngine:
    return PolicyEngine(PolicyConfig.model_validate(kwargs))


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (RiskLevel.READ_ONLY, Decision.ALLOW),
        (RiskLevel.LOW, Decision.ALLOW),
        (RiskLevel.MEDIUM, Decision.CONFIRM),
        (RiskLevel.HIGH, Decision.CONFIRM),
        (RiskLevel.CRITICAL, Decision.CONFIRM),
    ],
)
def test_default_thresholds(risk: RiskLevel, expected: Decision) -> None:
    assert engine().evaluate_tool("apps.open", risk).decision is expected


def test_lower_threshold_confirms_more() -> None:
    policy = engine(confirm_at_or_above="low")
    assert policy.evaluate_tool("apps.open", RiskLevel.LOW).decision is Decision.CONFIRM
    assert policy.evaluate_tool("apps.list", RiskLevel.READ_ONLY).decision is Decision.ALLOW


def test_high_risk_always_confirms_even_with_high_threshold() -> None:
    policy = engine(confirm_at_or_above="high")
    assert policy.evaluate_tool("apps.quit", RiskLevel.MEDIUM).decision is Decision.ALLOW
    assert policy.evaluate_tool("apps.quit", RiskLevel.HIGH).decision is Decision.CONFIRM
    assert policy.evaluate_tool("apps.quit", RiskLevel.CRITICAL).decision is Decision.CONFIRM


def test_tool_not_enabled_is_denied() -> None:
    policy = engine(enabled_tools=["apps.list"])
    result = policy.evaluate_tool("apps.open", RiskLevel.LOW)
    assert result.decision is Decision.DENY
    assert "enabled_tools" in result.reason


def test_blocklist_by_name_or_bundle_id_case_insensitive() -> None:
    policy = engine(app_blocklist=["terminal", "com.apple.Safari"])
    assert policy.check_app("Terminal", "com.apple.Terminal").decision is Decision.DENY
    assert policy.check_app("Safari", "com.apple.safari").decision is Decision.DENY
    assert policy.check_app("Notes", "com.apple.Notes").decision is Decision.ALLOW


def test_allowlist_restricts_to_listed_apps() -> None:
    policy = engine(app_allowlist=["Safari"])
    assert policy.check_app("Safari", None).decision is Decision.ALLOW
    assert policy.check_app("Notes", "com.apple.Notes").decision is Decision.DENY


def test_blocklist_wins_over_allowlist() -> None:
    policy = engine(app_allowlist=["Safari"], app_blocklist=["Safari"])
    assert policy.check_app("Safari", None).decision is Decision.DENY


def test_deny_all_confirmer() -> None:
    assert DenyAllConfirmer().confirm("anything") is False


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("YES", True), (" yes ", True), ("", False), ("n", False), ("sure", False)],
)
def test_interactive_confirmer_requires_explicit_yes(answer: str, expected: bool) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return answer

    confirmer = InteractiveConfirmer(input_fn=fake_input, is_interactive=lambda: True)
    assert confirmer.confirm("Quit Safari") is expected
    assert "Quit Safari" in prompts[0]


def test_interactive_confirmer_denies_without_terminal() -> None:
    def must_not_be_called(prompt: str) -> str:
        raise AssertionError("input() must not be called without a terminal")

    confirmer = InteractiveConfirmer(input_fn=must_not_be_called, is_interactive=lambda: False)
    assert confirmer.confirm("Quit Safari") is False


def test_interactive_confirmer_eof_is_no() -> None:
    def eof(prompt: str) -> str:
        raise EOFError

    assert InteractiveConfirmer(input_fn=eof, is_interactive=lambda: True).confirm("x") is False
