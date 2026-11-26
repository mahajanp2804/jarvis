"""Rule-based fast path: turn simple commands into tool calls without the AI model.

Simple, unambiguous commands ("open Safari", "quit Notes", "list apps") are
handled here in microseconds, so the language model (Phase 2) is only loaded
for requests these rules don't cover. The router never guesses: anything it
doesn't recognise returns ``None``.

Note: "run" and "start" are intentionally *not* treated as "open an app",
because "run my project" / "start the dev server" mean something else and will
be handled by the terminal tools and planner in later phases.
"""

from __future__ import annotations

import re
from datetime import datetime

from jarvis.api.schemas import ToolCall
from jarvis.errors import ToolInputError
from jarvis.tools.timers import mentions_later_day, parse_alarm_time, parse_duration

_POLITE_PREFIX = re.compile(
    r"^(?:(?:hey\s+)?jarvis[,:]?\s+)?(?:(?:please|can you|could you|would you)\s+)*",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"(?:\s+(?:please|for me|now))*[\s.!?]*$", re.IGNORECASE)
# Speech-to-text sometimes joins the verb to its object: "open-calculator."
_HYPHENATED_VERB = re.compile(r"^(open|launch|quit|close|exit)-(?=\w)", re.IGNORECASE)
_APP_SUFFIX = re.compile(r"\s+(?:app|application)$", re.IGNORECASE)
_LEADING_THE = re.compile(r"^the\s+", re.IGNORECASE)

_OPEN = re.compile(r"^(?:open|launch)\s+(?P<name>.+)$", re.IGNORECASE)
_QUIT = re.compile(r"^(?:quit|close|exit)\s+(?P<name>.+)$", re.IGNORECASE)
_LIST = re.compile(
    r"^(?:list|show|what are)(?:\s+(?:my|the|all|installed))*\s+"
    r"(?:apps|applications)(?:\s+(?:installed|i have))?"
    r"(?:\s+(?:matching|containing|named|with|like)\s+(?P<query>.+))?$",
    re.IGNORECASE,
)
# Targets that are clearly not application names (projects, files, websites).
# These fall through to the planner instead of failing as "app not found".
_NOT_AN_APP = re.compile(
    r"(?:^my\s)|[/\\~]|https?:|www\.|\.(?:com|org|net|io|dev|ai)\b|"
    r"\b(?:project|folder|directory|file|repo|repository|website|site|url|tab|window|document)s?\b",
    re.IGNORECASE,
)
# References that need conversation context ("quit it") go to the planner.
_PRONOUN = re.compile(
    r"^(?:it|that|this|them|those|these|that one|this one|everything|all(?: of them)?)$",
    re.IGNORECASE,
)

_FIND = re.compile(
    r"^(?:find|search(?:\s+for)?)\s+(?:an?\s+)?(?:apps?|applications?)\s+"
    r"(?:called\s+|named\s+|matching\s+)?(?P<query>.+)$",
    re.IGNORECASE,
)


# Memory: "remember that my editor is VS Code", "forget my editor", "what do you remember".
_REMEMBER = re.compile(
    r"^remember\s+(?:that\s+)?(?P<name>.+?)\s+(?:is|are)\s+(?P<value>.+)$", re.IGNORECASE
)
_FORGET = re.compile(r"^forget\s+(?:about\s+)?(?P<name>.+)$", re.IGNORECASE)
_RECALL = re.compile(
    r"^(?:what do you (?:remember|know)(?: about me)?|what are my preferences|"
    r"(?:list|show)(?: my)? (?:preferences|memory))$",
    re.IGNORECASE,
)
_MEMORY_NAME = re.compile(r"^(?:my|the)\s+", re.IGNORECASE)


# Everyday commands (Phase 8): exact phrasings only; anything else goes to the planner.
_F = re.IGNORECASE
_VOLUME_UP = re.compile(
    r"^(?:(?:turn|crank)\s+(?:the\s+)?volume\s+up|turn\s+up\s+the\s+(?:volume|sound)"
    r"|(?:increase|raise)\s+the\s+volume|volume\s+up|louder)$",
    _F,
)
_VOLUME_DOWN = re.compile(
    r"^(?:turn\s+(?:the\s+)?volume\s+down|turn\s+down\s+the\s+(?:volume|sound)"
    r"|(?:decrease|lower|reduce)\s+the\s+volume|volume\s+down|quieter|softer)$",
    _F,
)
_VOLUME_SET = re.compile(
    r"^(?:set\s+)?(?:the\s+)?volume\s+(?:to\s+)?(?P<level>\d{1,3})\s*(?:%|percent)?$", _F
)
_MUTE = re.compile(r"^(?P<un>un)?mute(?:\s+(?:the\s+)?(?:sound|volume|audio|mac))?$", _F)
_WIFI = re.compile(
    r"^(?:turn\s+(?P<a>on|off)\s+(?:the\s+)?wi-?fi|turn\s+(?:the\s+)?wi-?fi\s+(?P<b>on|off)"
    r"|wi-?fi\s+(?P<c>on|off))$",
    _F,
)
_BATTERY = re.compile(
    r"^(?:battery(?:\s+(?:level|status|percentage))?|how\s+much\s+battery"
    r"(?:\s+(?:do\s+i\s+have|is\s+left|have\s+i\s+got))?(?:\s+left)?"
    r"|what(?:'s|\s+is)\s+(?:my|the)\s+battery(?:\s+(?:level|at))?)$",
    _F,
)
_DISPLAY_OFF = re.compile(
    r"^(?:turn\s+off\s+the\s+(?:screen|display)|turn\s+the\s+(?:screen|display)\s+off"
    r"|sleep\s+the\s+(?:screen|display)|(?:screen|display)\s+off)$",
    _F,
)
_MUSIC = {
    "play": re.compile(r"^(?:play|resume|unpause)(?:\s+(?:the\s+)?music)?$", _F),
    "pause": re.compile(r"^pause(?:\s+(?:the\s+)?music)?$|^stop\s+(?:the\s+)?music$", _F),
    "next": re.compile(
        r"^(?:next(?:\s+(?:song|track))?|skip(?:\s+(?:this|the))?(?:\s+(?:song|track))?)$", _F
    ),
    "previous": re.compile(
        r"^(?:previous(?:\s+(?:song|track))?|go\s+back\s+a\s+(?:song|track)"
        r"|(?:play\s+)?the\s+last\s+(?:song|track)\s+again)$",
        _F,
    ),
}
_NOW_PLAYING = re.compile(
    r"^(?:what(?:'s|\s+is)\s+(?:this\s+song|playing|the\s+song)|what\s+song\s+is\s+"
    r"(?:this|playing))$",
    _F,
)
_PLAYLIST = re.compile(r"^play\s+(?:my\s+|the\s+)?(?P<name>.+?)\s+playlist$", _F)
_TIMER = re.compile(
    r"^(?:(?:set|start)\s+(?:a\s+)?timer\s+for\s+(?P<d1>.+?)|(?:set|start)\s+an?\s+"
    r"(?P<d2>.+?)\s+timer|timer\s+for\s+(?P<d3>.+?))(?:\s+(?:for|called)\s+(?P<label>.+))?$",
    _F,
)
_ALARM = re.compile(
    r"^(?:wake\s+me(?:\s+up)?\s+(?:at\s+)?|set\s+an?\s+alarm\s+(?:for\s+)?)(?P<time>.+)$", _F
)
_REMIND = re.compile(
    r"^remind\s+me\s+(?:to\s+(?P<what1>.+?)\s+(?P<when1>(?:at|in)\s+.+)"
    r"|(?P<when2>in\s+.+?)\s+to\s+(?P<what2>.+))$",
    _F,
)
_LIST_TIMERS = re.compile(
    r"^(?:(?:list|show)\s+(?:my\s+|the\s+|all\s+)?(?:timers|alarms|reminders|timers\s+and"
    r"\s+alarms)|what\s+timers\s+do\s+i\s+have|how\s+(?:much\s+time|long)\s+(?:is\s+)?"
    r"left\s+on\s+(?:my|the)\s+timer)$",
    _F,
)
_CANCEL_TIMER = re.compile(
    r"^(?:cancel|stop|delete)\s+(?P<which>(?:the\s+|my\s+)?(?:all\s+)?(?:timers?|alarms?)"
    r"|(?:the\s+|my\s+)?\S+\s+timer)$",
    _F,
)
_WEATHER = re.compile(
    r"^(?:what(?:'s|\s+is)\s+the\s+)?weather(?:\s+like)?(?:\s+(?:in|for)\s+(?P<place>.+?))?"
    r"(?:\s+(?:today|tomorrow|this\s+week|right\s+now|now))?$",
    _F,
)


def _route_everyday(command: str) -> ToolCall | None:
    """Volume, Wi-Fi, battery, display, music, timers and weather; None if not recognised."""

    def call(tool: str, **arguments: object) -> ToolCall:
        return ToolCall(tool=tool, arguments=arguments)

    if _VOLUME_UP.match(command):
        return call("system.volume", change="up")
    if _VOLUME_DOWN.match(command):
        return call("system.volume", change="down")
    if (match := _VOLUME_SET.match(command)) and int(match.group("level")) <= 100:
        return call("system.volume", level=int(match.group("level")))
    if match := _MUTE.match(command):
        return call("system.volume", mute=not match.group("un"))
    if match := _WIFI.match(command):
        state = (match.group("a") or match.group("b") or match.group("c")).lower()
        return call("system.wifi", on=state == "on")
    if _BATTERY.match(command):
        return call("system.status")
    if _DISPLAY_OFF.match(command):
        return call("system.sleep_display")
    for name, pattern in _MUSIC.items():
        if pattern.match(command):
            return call("music.control", command=name)
    if _NOW_PLAYING.match(command):
        return call("music.status")
    if match := _PLAYLIST.match(command):
        return call("music.play", playlist=match.group("name").strip(" \"'"))
    if match := _TIMER.match(command):
        duration = match.group("d1") or match.group("d2") or match.group("d3")
        if _is_duration(duration):
            label = (match.group("label") or "").strip(" \"'")
            return call("timers.start", duration=duration, **({"label": label} if label else {}))
        return None
    if mentions_later_day(command) and re.match(r"^(?:remind|wake|set\s+an?\s+alarm)", command, _F):
        return None  # only today or tomorrow: the planner explains
    if match := _REMIND.match(command):
        what = (match.group("what1") or match.group("what2")).strip(" \"'")
        when = match.group("when1") or match.group("when2")
        when = when.removeprefix("at ").strip()
        return call("timers.alarm", time=when, label=what) if _is_time(when) else None
    if match := _ALARM.match(command):
        time = match.group("time").strip()
        return call("timers.alarm", time=time) if _is_time(time) else None
    if _LIST_TIMERS.match(command):
        return call("timers.list")
    if match := _CANCEL_TIMER.match(command):
        return call("timers.cancel", which=match.group("which"))
    if match := _WEATHER.match(command):
        place = (match.group("place") or "").strip(" \"'")
        return call("weather.forecast", **({"location": place} if place else {}))
    return None


def _is_duration(text: str) -> bool:
    try:
        parse_duration(text)
    except ToolInputError:
        return False
    return True


def _is_time(text: str) -> bool:
    try:
        parse_alarm_time(text, datetime.now().astimezone())
    except ToolInputError:
        return False
    return True


def _memory_name(raw: str) -> str | None:
    name = _MEMORY_NAME.sub("", raw.strip())
    return None if not name or _PRONOUN.match(name) or name.lower() in {"i", "you"} else name


def _clean_app_name(raw: str) -> str | None:
    """Strip filler words; return ``None`` if the target is not an application."""
    if _NOT_AN_APP.search(raw) or _PRONOUN.match(raw.strip()):
        return None
    name = _LEADING_THE.sub("", raw.strip())
    name = _APP_SUFFIX.sub("", name)
    return name.strip(" \"'") or None


def normalise_command(text: str) -> str:
    """Collapse spaces and strip "hey Jarvis, please" / "for me." from a request."""
    command = " ".join(text.split())
    command = _POLITE_PREFIX.sub("", command)
    command = _HYPHENATED_VERB.sub(r"\1 ", command)
    return _TRAILING.sub("", command).strip()


def route(text: str) -> ToolCall | None:
    """Map a simple command to a :class:`ToolCall`, or ``None`` if not recognised."""
    command = normalise_command(text)
    if not command:
        return None

    if match := _LIST.match(command):
        query = match.group("query")
        args = {"query": query.strip(" \"'")} if query else {}
        return ToolCall(tool="apps.list", arguments=args)

    if match := _FIND.match(command):
        return ToolCall(tool="apps.list", arguments={"query": match.group("query").strip(" \"'")})

    if _RECALL.match(command):
        return ToolCall(tool="memory.list")

    if match := _REMEMBER.match(command):
        name = _memory_name(match.group("name"))
        value = match.group("value").strip(" \"'")
        if name and value:
            return ToolCall(tool="memory.remember", arguments={"name": name, "value": value})
        return None

    if match := _FORGET.match(command):
        name = _memory_name(match.group("name"))
        return ToolCall(tool="memory.forget", arguments={"name": name}) if name else None

    if everyday := _route_everyday(command):
        return everyday

    if match := _OPEN.match(command):
        name = _clean_app_name(match.group("name"))
        return ToolCall(tool="apps.open", arguments={"name": name}) if name else None

    if match := _QUIT.match(command):
        name = _clean_app_name(match.group("name"))
        return ToolCall(tool="apps.quit", arguments={"name": name}) if name else None

    return None
