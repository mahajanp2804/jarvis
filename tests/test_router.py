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
        ("open-calculator.", "apps.open", {"name": "calculator"}),  # as transcribed
        ("Hey Jarvis, open Google Chrome.", "apps.open", {"name": "Google Chrome"}),
        ("quit Music", "apps.quit", {"name": "Music"}),
        ("Quit-Music", "apps.quit", {"name": "Music"}),
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
    ("text", "tool", "arguments"),
    [
        ("turn up the volume", "system.volume", {"change": "up"}),
        ("Hey Jarvis, turn the volume down please.", "system.volume", {"change": "down"}),
        ("set the volume to 30%", "system.volume", {"level": 30}),
        ("mute", "system.volume", {"mute": True}),
        ("unmute the sound", "system.volume", {"mute": False}),
        ("turn off wifi", "system.wifi", {"on": False}),
        ("turn the Wi-Fi on", "system.wifi", {"on": True}),
        ("how much battery do I have left?", "system.status", {}),
        ("turn off the screen", "system.sleep_display", {}),
        ("pause the music", "music.control", {"command": "pause"}),
        ("play", "music.control", {"command": "play"}),
        ("skip this song", "music.control", {"command": "next"}),
        ("previous track", "music.control", {"command": "previous"}),
        ("what's playing?", "music.status", {}),
        ("play my road trip playlist", "music.play", {"playlist": "road trip"}),
        ("set a timer for 10 minutes", "timers.start", {"duration": "10 minutes"}),
        ("set a 5 minute timer", "timers.start", {"duration": "5 minute"}),
        ("set a timer for 10 minutes for pasta", "timers.start",
         {"duration": "10 minutes", "label": "pasta"}),
        ("wake me up at 7 am", "timers.alarm", {"time": "7 am"}),
        ("set an alarm for 6:30 tomorrow", "timers.alarm", {"time": "6:30 tomorrow"}),
        ("remind me to buy milk at 6 pm", "timers.alarm", {"time": "6 pm", "label": "buy milk"}),
        ("remind me in 20 minutes to stretch", "timers.alarm",
         {"time": "in 20 minutes", "label": "stretch"}),
        ("how long is left on my timer?", "timers.list", {}),
        ("cancel the pasta timer", "timers.cancel", {"which": "the pasta timer"}),
        ("cancel all alarms", "timers.cancel", {"which": "all alarms"}),
        ("what's the weather?", "weather.forecast", {}),
        ("what's the weather like in New York today", "weather.forecast", {"location": "New York"}),
    ],
)  # fmt: skip
def test_routes_everyday_commands(text: str, tool: str, arguments: dict[str, object]) -> None:
    assert route(text) == ToolCall(tool=tool, arguments=arguments)


@pytest.mark.parametrize(
    "text",
    [
        "play some jazz",  # a search: the planner picks the words
        "stop",  # stop what?
        "stop p1",
        "set a timer",  # for how long?
        "set a timer for later",
        "wake me up when it's done",
        "remind me later",
        "remind me to call mom on friday at 5",  # only today or tomorrow
        "set the volume to 150",
        "turn the volume up to eleven",
    ],
)
def test_everyday_commands_never_guess(text: str) -> None:
    assert route(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello",
        "open",
        "what's the weather going to do?",
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
