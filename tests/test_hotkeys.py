from __future__ import annotations

import pytest

from jarvis.platforms.macos_gui import carbon_codes
from jarvis.runtime.hotkeys import Hotkey, HotkeyError


@pytest.mark.parametrize(
    ("text", "canonical", "symbol"),
    [
        ("ctrl+option+cmd+j", "ctrl+option+cmd+j", "⌃⌥⌘J"),
        ("Cmd + Shift + Space", "shift+cmd+space", "⇧⌘Space"),
        ("⌃⌥⌘J", "ctrl+option+cmd+j", "⌃⌥⌘J"),
        ("⌘⇧ + f5", "shift+cmd+f5", "⇧⌘F5"),
        ("control+alt+1", "ctrl+option+1", "⌃⌥1"),
    ],
)
def test_parse(text: str, canonical: str, symbol: str) -> None:
    hotkey = Hotkey.parse(text)
    assert str(hotkey) == canonical
    assert hotkey.symbol() == symbol
    assert Hotkey.parse(str(hotkey)) == hotkey  # round trip


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("j", "Invalid shortcut"),
        ("cmd+", "Invalid shortcut"),
        ("option+space", "Command or Control"),
        ("shift+option+k", "Command or Control"),
        ("hyper+j", "Unknown modifier"),
        ("cmd+ü", "Unsupported key"),
        ("cmd+f13", "Unsupported key"),
    ],
)
def test_invalid(text: str, match: str) -> None:
    with pytest.raises(HotkeyError, match=match):
        Hotkey.parse(text)


def test_carbon_codes() -> None:
    # kVK_ANSI_J = 38; cmdKey 256 | optionKey 2048 | controlKey 4096
    assert carbon_codes(Hotkey.parse("ctrl+option+cmd+j")) == (38, 256 | 2048 | 4096)
    assert carbon_codes(Hotkey.parse("cmd+shift+space")) == (49, 256 | 512)
    assert carbon_codes(Hotkey.parse("cmd+f12")) == (111, 256)


def test_every_supported_key_has_a_mac_key_code() -> None:
    from jarvis.runtime.hotkeys import KEYS

    for key in KEYS:
        carbon_codes(Hotkey(frozenset({"cmd"}), key))
