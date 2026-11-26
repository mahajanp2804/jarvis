"""Pick which tool groups the model sees for a request.

Every tool in the system prompt costs prompt tokens (slower first requests,
less room in the context window) and gives a small model more ways to pick
the wrong tool. So a request about files sees the file tools, not the
terminal or browser ones.

The choice is one of a few fixed *profiles* rather than any combination of
groups: the system prompt's saved state (see :mod:`jarvis.ai.prompt_cache`)
is keyed by the exact prompt, so a handful of stable prompts stay cached
while per-request subsets would mean slow, uncached first reads.

Groups are chosen by keywords, never by the model. When nothing matches
(chat, requests Jarvis has no tool for) or a request needs a mix no profile
covers, the model sees every enabled tool, as before.
"""

from __future__ import annotations

import re

# A tool's group is the part of its name before the dot: "files.read" -> "files".
_KEYWORDS: dict[str, re.Pattern[str]] = {
    group: re.compile(pattern, re.IGNORECASE)
    for group, pattern in {
        "apps": r"\b(?:open|launch|quit|close|start|apps?|applications?|installed)\b",
        "files": (
            r"~/|\b(?:files?|folders?|documents?|desktop|downloads|pdfs?|read|rename|move"
            r"|trash|delete|organi[sz]e|sort|create|make|resume|invoice|report)\b"
            r"|\w\.(?:txt|md|pdf|csv|docx?|xlsx?|pages|numbers|json|py)\b"
        ),
        "projects": r"\b(?:projects?|repo|repository|codebase)\b",
        "terminal": (
            r"\b(?:run|tests?|git|npm|pip|python3?|node|server|dev|build|install|terminal"
            r"|process(?:es)?|commands?|p\d+|output|requirements)\b"
        ),
        "browser": (
            r"\b(?:website|internet|online|google|bing|https?|url|youtube|wikipedia|browser"
            r"|search the web|web search)\b|\w\.(?:com|org|net|io|dev|in|co)\b"
        ),
        "memory": r"\b(?:remember|forget|memory|preferences?|earlier|history)\b",
        # Words that suggest the web but also appear in project and folder names
        # ("Stock Price Predictor", "~/Code/web"): ignored when the request is
        # about a project or the terminal.
        "browser_weak": r"\b(?:web|site|search|look up|news|weather|price|page|recipe)\b",
        "screen": r"\b(?:screen|screenshot)\b",
        "shortcuts": r"\bshortcuts?\b",
    }.items()
}

# Smallest first; each is a stable system prompt. Memory is tiny and always included.
PROFILES: tuple[tuple[str, frozenset[str]], ...] = (
    ("apps", frozenset({"apps", "memory"})),
    ("files", frozenset({"apps", "files", "memory"})),
    ("web", frozenset({"apps", "browser", "files", "memory"})),
    ("dev", frozenset({"apps", "files", "projects", "terminal", "memory"})),
)


def wanted_groups(request: str) -> set[str]:
    """Groups whose keywords appear in ``request``."""
    found = {group for group, pattern in _KEYWORDS.items() if pattern.search(request)}
    if "browser_weak" in found:
        found.discard("browser_weak")
        if not found & {"projects", "terminal"}:
            found.add("browser")
    return found


def select_profile(request: str) -> tuple[str, frozenset[str] | None]:
    """The profile for ``request``: (name, groups), or ("all", None) for every tool."""
    wanted = wanted_groups(request)
    if wanted:
        for name, groups in PROFILES:
            if wanted <= groups:
                return name, groups
    return "all", None


def group_of(tool_name: str) -> str:
    return tool_name.split(".", 1)[0]
