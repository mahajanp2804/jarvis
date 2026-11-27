"""The Contacts app on macOS, through a fixed AppleScript.

Finds people by name or nickname, and through the related names on your own
card ("mother", "spouse"), so "text mom" can find the right person. The
search words are passed as ``argv``. Labels come back either plain ("mobile")
or in the app's internal form ("_$!<Mobile>!$_"), which is tidied here.
"""

from __future__ import annotations

import re
import subprocess

from jarvis.platforms.applescript import CLEAN, AppleScript, Runner, records
from jarvis.platforms.base import Contact, ContactsBook

FIND = f"""\
{CLEAN}
on handles(items_)
	set out to ""
	repeat with x in items_
		set out to out & my clean(label of x) & "=" & my clean(value of x) & ";"
	end repeat
	return out
end handles
on run argv
	set q to item 1 of argv
	set rel to item 2 of argv
	set out to ""
	tell application "Contacts"
		set found to (every person whose name contains q or nickname contains q)
		if rel is not "" then
			try
				repeat with r in (related names of my card)
					if (label of r) contains rel then
						set found to found & (every person whose name is (value of r))
					end if
				end repeat
			end try
		end if
		repeat with p in found
			set row to my clean(id of p) & tab & my clean(name of p)
			set row to row & tab & my handles(phones of p) & tab & my handles(emails of p)
			set out to out & row & linefeed
		end repeat
	end tell
	return out
end run"""

ALL_SCRIPTS = (FIND,)

_INTERNAL_LABEL = re.compile(r"^_\$!<(.+)>!\$_$")


def tidy_label(label: str) -> str:
    """ "_$!<Mobile>!$_" -> "mobile"."""
    match = _INTERNAL_LABEL.match(label.strip())
    return (match.group(1) if match else label.strip()).lower()


def _handles(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in text.split(";"):
        label, _, value = item.rpartition("=")  # no "=": a value without a label
        if value.strip():
            pairs.append((tidy_label(label), value.strip()))
    return tuple(pairs)


class MacContacts(ContactsBook):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._script = AppleScript("Contacts", runner)

    def find(self, name: str, relation: str | None = None) -> list[Contact]:
        found = self._script(FIND, f"look up “{name}” in Contacts", name, relation or "")
        people: dict[str, Contact] = {}
        for pid, full_name, phones, emails in records(found, 4):
            people.setdefault(pid, Contact(pid, full_name, _handles(phones), _handles(emails)))
        return list(people.values())
