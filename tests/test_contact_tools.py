"""contacts.find and recipient lookup on the fake address book."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from jarvis.api.schemas import ToolCall
from jarvis.bootstrap import Runtime
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.tools.contacts import Recipient, is_phone, resolve_recipient
from tests.fakes import FakeContacts, FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def test_find(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    runtime, confirmer = make_runtime()
    result = runtime.dispatcher.execute(ToolCall(tool="contacts.find", arguments={"name": "mom"}))
    assert result.message == (
        "Contacts matching “mom” (1):\n"
        "  Jane Doe: mobile +1 555 0100, home +1 555 0199, home jane@example.com"
    )
    assert result.untrusted and confirmer.prompts == []
    assert fake_platform.contact_book.queries == [("mom", "mother")]


def test_numbers_and_addresses_are_used_as_given() -> None:
    book = FakeContacts()
    assert resolve_recipient(book, "+1 (555) 010-0100", "text") == Recipient(
        "+1 (555) 010-0100", "+1 (555) 010-0100"
    )
    assert resolve_recipient(book, "a@b.co", "email").handle == "a@b.co"
    assert book.queries == []
    assert not is_phone("12345") and not is_phone("tomorrow at 5")


def test_names_and_relations() -> None:
    book = FakeContacts()
    assert resolve_recipient(book, "my mom", "text") == Recipient("Jane Doe", "+1 555 0100")
    assert resolve_recipient(book, "jane doe", "email").handle == "jane@example.com"
    assert resolve_recipient(book, "Sam Patel", "text").handle == "+1 555 0177"


def test_unclear_recipients_ask() -> None:
    book = FakeContacts()
    with pytest.raises(NeedsClarificationError, match="Which Sam: Sam Lee, Sam Patel"):
        resolve_recipient(book, "Sam", "text")
    book.people[0] = book.people[0].__class__(
        "p1", "Jane Doe", (("home", "+1 555 0199"), ("work", "+1 555 0123")), ()
    )
    with pytest.raises(NeedsClarificationError, match="Which one for Jane Doe: home"):
        resolve_recipient(book, "mom", "text")
    with pytest.raises(ToolExecutionError, match="no email address"):
        resolve_recipient(book, "Sam Patel", "email")
    with pytest.raises(ToolExecutionError, match="no “Bob” in your contacts"):
        resolve_recipient(book, "Bob", "text")
