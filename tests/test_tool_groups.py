from __future__ import annotations

import pytest

from jarvis.agent.tool_groups import group_of, select_profile
from jarvis.evaluation import load_cases


@pytest.mark.parametrize(
    ("request_text", "profile"),
    [
        ("open safari", "apps"),
        ("which apps do I have for editing photos?", "apps"),
        ("remember that my editor is Zed", "apps"),
        ("what's on the todo list on my desktop?", "files"),
        ("rename ~/Desktop/report.txt to final-report.txt", "files"),
        ("search the web for llama.cpp", "web"),
        ("open github.com", "web"),
        ("open my Stock Price Predictor project and run it", "dev"),
        ("stop p1", "dev"),
        ("turn the volume down", "system"),
        ("how much battery do I have left?", "system"),
        ("turn off wifi", "system"),
        ("run my water eject shortcut", "system"),
        ("run the tests, then run my Notify shortcut", "all"),
        # No keywords, or a mix no profile covers: every tool, as before.
        ("set a timer for 10 minutes", "all"),
        ("hi jarvis", "all"),
        ("find my weather app project", "dev"),  # "weather" is a project name here
        ("start the dev server in ~/Desktop/Code/web", "dev"),
        ("what's the weather in mumbai today?", "web"),
        ("take a screenshot", "all"),  # screen tools are opt-in, in no profile
    ],
)
def test_profiles(request_text: str, profile: str) -> None:
    assert select_profile(request_text)[0] == profile


def test_routing_never_hides_a_tool_the_test_set_expects() -> None:
    """For every case, some acceptable tool list is fully visible in the chosen profile."""
    _, cases = load_cases()
    for case in cases:
        name, groups = select_profile(case.request)
        if groups is None or not case.tools:
            continue
        assert any({group_of(t) for t in tools} <= groups for tools in case.tools), (
            f"{case.id}: profile {name} {sorted(groups)} hides {case.tools}"
        )


def test_most_requests_get_a_smaller_profile() -> None:
    _, cases = load_cases()
    narrowed = [c for c in cases if c.tools and select_profile(c.request)[1] is not None]
    executes = [c for c in cases if c.tools]
    assert len(narrowed) / len(executes) > 0.8
