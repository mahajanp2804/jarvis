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

from jarvis.api.schemas import ToolCall

_POLITE_PREFIX = re.compile(
    r"^(?:(?:hey\s+)?jarvis[,:]?\s+)?(?:(?:please|can you|could you|would you)\s+)*",
    re.IGNORECASE,
)
_TRAILING = re.compile(r"(?:\s+(?:please|for me|now))*[\s.!?]*$", re.IGNORECASE)
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

    if match := _OPEN.match(command):
        name = _clean_app_name(match.group("name"))
        return ToolCall(tool="apps.open", arguments={"name": name}) if name else None

    if match := _QUIT.match(command):
        name = _clean_app_name(match.group("name"))
        return ToolCall(tool="apps.quit", arguments={"name": name}) if name else None

    return None
