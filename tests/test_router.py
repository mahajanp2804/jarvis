from __future__ import annotations

import pytest

from jarvis.agent.router import route
from jarvis.api.schemas import ToolCall


@pytest.mark.parametrize(
    ("text", "tool", "arguments"),
    [
        ("open Safari", "apps.open", {"name": "Safari"}),
        ("Open safari.", "apps.open", {"name": "safari"}),
        ("launch the Notes app", "apps.open", {"name": "Notes"}),
        (
            "Jarvis, please open Visual Studio Code for me",
            "apps.open",
            {"name": "Visual Studio Code"},
        ),
        ("hey jarvis can you open   photo booth please!", "apps.open", {"name": "photo booth"}),
        ("open 'Activity Monitor'", "apps.open", {"name": "Activity Monitor"}),
        ("quit Music", "apps.quit", {"name": "Music"}),
        ("close the Calculator application", "apps.quit", {"name": "Calculator"}),
        ("list apps", "apps.list", {}),
        ("show all installed applications", "apps.list", {}),
        ("what are the apps i have", "apps.list", {}),
        ("list apps matching studio", "apps.list", {"query": "studio"}),
        ("find app code", "apps.list", {"query": "code"}),
        ("search for an application named xcode", "apps.list", {"query": "xcode"}),
    ],
)
def test_routes_simple_commands(text: str, tool: str, arguments: dict[str, str]) -> None:
    assert route(text) == ToolCall(tool=tool, arguments=arguments)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello",
        "open",
        "what's the weather?",
        # "run"/"start" are reserved for projects and commands (later phases)
        "run Safari",
        "start the dev server",
        # targets that are not applications go to the planner, not the app tool
        "open my Stock Price Predictor project and run it",
        "open ~/Downloads",
        "open https://example.com",
        "open github.com",
        "open the downloads folder",
        "close this window",
        "open resume.pdf file",
        # pronouns need conversation context, which only the planner has
        "quit it",
        "open that one",
        "close them",
        "quit all of them",
    ],
)
def test_does_not_guess(text: str) -> None:
    assert route(text) is None
