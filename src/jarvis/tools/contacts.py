"""Contacts: look someone up, and work out who "mom" or "Sam" is for a message.

* ``contacts.find`` is read-only. Names, numbers and addresses are shown as
  outside content (cards can come from other people).
* :func:`resolve_recipient` is used by the Messages and Mail tools: a number
  or address is used as given; a name or a relation ("mom", "my wife") is
  looked up, and when several people or several numbers match, Jarvis asks
  which one instead of guessing. Texts prefer a mobile number.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.platforms.base import Contact, ContactsBook, PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec

# Words for people, mapped to the relation labels on your own contact card.
RELATIONS = {
    "mom": "mother", "mum": "mother", "mommy": "mother", "mama": "mother", "mother": "mother",
    "dad": "father", "daddy": "father", "papa": "father", "father": "father",
    "wife": "spouse", "husband": "spouse", "spouse": "spouse", "partner": "partner",
    "brother": "brother", "sister": "sister", "son": "son", "daughter": "daughter",
    "boss": "manager", "manager": "manager", "assistant": "assistant",
}  # fmt: skip
_PHONE = re.compile(r"^\+?[\d\s().-]{7,20}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MOBILE_LABELS = {"mobile", "iphone", "cell", "main"}
MAX_LISTED = 10


@dataclass(frozen=True)
class Recipient:
    name: str  # the contact's name, or the number/address when given directly
    handle: str  # the number or address used

    def describe(self) -> str:
        return self.handle if self.name == self.handle else f"{self.name} ({self.handle})"


def is_phone(text: str) -> bool:
    return bool(_PHONE.match(text.strip())) and sum(c.isdigit() for c in text) >= 7


def is_email(text: str) -> bool:
    return bool(_EMAIL.match(text.strip()))


def lookup(book: ContactsBook, who: str) -> list[Contact]:
    words = " ".join(who.split()).removeprefix("my ").removeprefix("My ").strip()
    relation = RELATIONS.get(words.lower())
    people = book.find(words, relation)
    exact = [p for p in people if p.name.casefold() == words.casefold()]
    return exact or people


def resolve_recipient(book: ContactsBook, to: str, kind: Literal["text", "email"]) -> Recipient:
    """Who to send to. Raises NeedsClarificationError when it isn't clear."""
    text = " ".join(to.split())
    if is_email(text) or (kind == "text" and is_phone(text)):
        return Recipient(text, text)
    people = lookup(book, text)
    if not people:
        raise ToolExecutionError(
            f"There's no “{text}” in your contacts.",
            hint="Use the name as it is in Contacts, or give the number or address.",
        )
    if len(people) > 1:
        names = ", ".join(p.name for p in people[:6])
        raise NeedsClarificationError(f"Which {text}: {names}?")
    person = people[0]
    if kind == "text":
        mobiles = [h for h in person.phones if h[0] in _MOBILE_LABELS]
        handles = mobiles or list(person.phones) or list(person.emails)
    else:
        handles = list(person.emails)
    if not handles:
        what = "phone number or address" if kind == "text" else "email address"
        raise ToolExecutionError(f"{person.name} has no {what} in Contacts.")
    if len(handles) > 1:
        options = ", ".join(f"{label or 'other'} {value}" for label, value in handles)
        raise NeedsClarificationError(f"Which one for {person.name}: {options}?")
    return Recipient(person.name, handles[0][1])


def describe_contact(person: Contact) -> str:
    details = [f"{label or 'phone'} {value}" for label, value in person.phones]
    details += [f"{label or 'email'} {value}" for label, value in person.emails]
    return f"{person.name}: " + (", ".join(details) if details else "no number or address")


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80, description='a name, or e.g. "mom"')


class ContactTools:
    """Builds the ``contacts.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "contacts.find",
                "Look up a person's phone numbers and email addresses in Contacts.",
                FindArgs,
                RiskLevel.READ_ONLY,
                self._prepare_find,
                idempotent=True,
            )
        ]

    def _prepare_find(self, args: FindArgs) -> PreparedAction:
        book = self._platform.contacts()  # UnsupportedPlatformError off macOS
        who = " ".join(args.name.split())

        def run(cancel: threading.Event) -> ToolResult:
            people = lookup(book, who)
            if not people:
                message = f"No one called “{who}” in your contacts."
            else:
                lines = [f"  {describe_contact(p)}" for p in people[:MAX_LISTED]]
                more = (
                    f"\n  … and {len(people) - MAX_LISTED} more" if len(people) > MAX_LISTED else ""
                )
                message = f"Contacts matching “{who}” ({len(people)}):\n" + "\n".join(lines) + more
            return ToolResult(
                tool="contacts.find",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"count": len(people)},
                untrusted=bool(people),
            )

        return PreparedAction(description=f"Look up “{who}” in Contacts", run=run)
