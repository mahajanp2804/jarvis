"""Spoken replies: the speech-output contract and what gets said.

Speaking is off by default (``[voice] speak_replies``). When on, Jarvis says a
short summary of each reply (the answer, or the outcome of the last step), not
the whole text: lists, file contents and web pages are shown, not read aloud.
Speech runs on-device (``say`` on macOS), starts without blocking, and stops
when a new request starts, when you press the shortcut again, and when Jarvis
stops.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.agent.orchestrator import Reply

DEFAULT_SPOKEN_CHARS = 300

_MARKS = re.compile(r"^[\s✓✗•…*-]+")
_URL = re.compile(r"https?://([^/\s:]+)(?::(\d+))?[^\s]*")
_CODE = re.compile(r"`([^`]*)`")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Voice:
    name: str
    language: str


class SpeechOutput(ABC):
    """Text-to-speech that runs in the background."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Start saying ``text`` (stopping anything still being said). Returns at once."""

    @abstractmethod
    def stop(self) -> None:
        """Stop speaking now (no-op when silent)."""

    @property
    @abstractmethod
    def speaking(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool:
        """Block until speech ends. True if it ended within ``timeout``."""
        return True

    def check(self) -> None:  # noqa: B027 - optional hook; most speakers have nothing to check
        """Raise :class:`~jarvis.errors.JarvisError` if speech can't work as configured."""

    def voices(self) -> list[Voice]:
        return []


def clean_for_speech(text: str, limit: int = DEFAULT_SPOKEN_CHARS) -> str:
    """One line of text made pleasant to hear: no marks, code quotes or long URLs."""
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    line = _MARKS.sub("", line)
    line = _URL.sub(lambda m: m.group(1) + (f" port {m.group(2)}" if m.group(2) else ""), line)
    line = _CODE.sub(r"\1", line).replace("~/", "")
    line = _SPACES.sub(" ", line).strip()
    if len(line) <= limit:
        return line
    cut = line[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= limit // 4:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "…"


def spoken_text(reply: Reply, limit: int = DEFAULT_SPOKEN_CHARS) -> str:
    """What to say for a reply: the answer or question, else the last step's outcome."""
    from jarvis.agent.planner import PlanAction

    if reply.stopped:
        last = reply.results[-1].message if reply.results else ""
        return clean_for_speech(f"{clean_for_speech(last, limit)} {reply.stopped}", limit)
    if reply.plan is not None and reply.plan.action in (PlanAction.RESPOND, PlanAction.CLARIFY):
        return clean_for_speech(reply.plan.message, limit)
    if reply.results:
        return clean_for_speech(reply.results[-1].message, limit)
    return clean_for_speech(reply.text, limit)
