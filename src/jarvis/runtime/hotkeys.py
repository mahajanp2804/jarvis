"""Global keyboard shortcut: parsing, validation, and the listener interface.

Shortcuts are written like ``"ctrl+option+cmd+j"``. Parsing and validation are
platform-neutral; each OS adapter maps the result to native key codes.

Shortcuts must include Command or Control. macOS 15+ no longer delivers
Option-only global shortcuts, and requiring a "command" modifier keeps a
shortcut from swallowing ordinary typing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from jarvis.errors import JarvisError

MODIFIER_ALIASES = {
    "cmd": "cmd",
    "command": "cmd",
    "⌘": "cmd",
    "ctrl": "ctrl",
    "control": "ctrl",
    "⌃": "ctrl",
    "opt": "option",
    "option": "option",
    "alt": "option",
    "⌥": "option",
    "shift": "shift",
    "⇧": "shift",
}
MODIFIER_ORDER = ("ctrl", "option", "shift", "cmd")
SYMBOLS = {"ctrl": "⌃", "option": "⌥", "shift": "⇧", "cmd": "⌘"}

KEYS = frozenset(
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
    + ["space", "return", "tab", "escape"]
    + [f"f{n}" for n in range(1, 13)]
)


class HotkeyError(JarvisError):
    """The shortcut is invalid or could not be registered."""


@dataclass(frozen=True)
class Hotkey:
    modifiers: frozenset[str]
    key: str

    @classmethod
    def parse(cls, text: str) -> Hotkey:
        normalised = "".join(text.split()).lower()
        for symbol in SYMBOLS.values():  # "⌘⇧space" -> "⌘+⇧+space"
            normalised = normalised.replace(symbol, symbol + "+")
        while "++" in normalised:
            normalised = normalised.replace("++", "+")
        parts = normalised.split("+")
        if len(parts) < 2 or not all(parts):
            raise HotkeyError(
                f"Invalid shortcut '{text}'.",
                hint='Use modifiers and one key joined by "+", e.g. "ctrl+option+cmd+j".',
            )
        *mods, key = parts
        unknown = [m for m in mods if m not in MODIFIER_ALIASES]
        if unknown:
            raise HotkeyError(f"Unknown modifier(s) in '{text}': {', '.join(unknown)}.")
        if key not in KEYS:
            raise HotkeyError(
                f"Unsupported key '{key}' in '{text}'.",
                hint="Use a letter, digit, space, return, tab, escape or f1-f12.",
            )
        modifiers = frozenset(MODIFIER_ALIASES[m] for m in mods)
        if not modifiers & {"cmd", "ctrl"}:
            raise HotkeyError(
                f"Shortcut '{text}' must include Command or Control.",
                hint="macOS does not deliver Option-only global shortcuts.",
            )
        return cls(modifiers, key)

    def __str__(self) -> str:
        return "+".join([m for m in MODIFIER_ORDER if m in self.modifiers] + [self.key])

    def symbol(self) -> str:
        """macOS-style display, e.g. ``⌃⌥⌘J``."""
        mods = "".join(SYMBOLS[m] for m in MODIFIER_ORDER if m in self.modifiers)
        return mods + (self.key.upper() if len(self.key) == 1 else self.key.capitalize())


class HotkeyListener(ABC):
    """Registers one global shortcut and calls back on the event-loop thread."""

    @abstractmethod
    def register(self, hotkey: Hotkey, callback: Callable[[], None]) -> None:
        """Start listening. Raise :class:`HotkeyError` if the OS refuses."""

    @abstractmethod
    def unregister(self) -> None:
        """Stop listening (safe to call more than once)."""
