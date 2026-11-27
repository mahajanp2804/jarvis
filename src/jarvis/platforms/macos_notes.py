"""The Notes app on macOS, through fixed AppleScripts.

* Search matches the title or the text; locked (password-protected) notes are
  left out, since their text can't be read.
* A new note's body is HTML inside Notes: the title and text are escaped here
  and passed as ``argv``, so nothing in them is read as markup or script.
"""

from __future__ import annotations

import html
import subprocess
from datetime import datetime

from jarvis.errors import ToolExecutionError
from jarvis.platforms.applescript import CLEAN, AppleScript, Runner, records
from jarvis.platforms.base import Note, NotesApp
from jarvis.platforms.macos_reminders import DATES

EXCERPT = 160
LOCKED = "#locked"

SEARCH = f"""\
{CLEAN}
{DATES}
on run argv
	set q to item 1 of argv
	set n to (item 2 of argv) as integer
	set out to ""
	tell application "Notes"
		set found to (every note whose name contains q or plaintext contains q)
		set shown to 0
		repeat with x in found
			if shown ≥ n then exit repeat
			if not (password protected of x) then
				set excerpt to plaintext of x
				if (length of excerpt) > {EXCERPT} then
					set excerpt to text 1 thru {EXCERPT} of excerpt
				end if
				set row to my clean(id of x) & tab & my clean(name of x)
				set row to row & tab & my clean(name of (container of x))
				set row to row & tab & my iso(modification date of x)
				set out to out & row & tab & my clean(excerpt) & linefeed
				set shown to shown + 1
			end if
		end repeat
	end tell
	return out
end run"""

READ = f"""\
{CLEAN}
{DATES}
on run argv
	tell application "Notes"
		set x to note id (item 1 of argv)
		if password protected of x then return "{LOCKED}"
		set head to my clean(id of x) & tab & my clean(name of x)
		set head to head & tab & my clean(name of (container of x))
		set head to head & tab & my iso(modification date of x)
		return head & linefeed & (plaintext of x)
	end tell
end run"""

CREATE = f"""\
{CLEAN}
on run argv
	set {{theBody, folderName}} to items 1 thru 2 of argv
	tell application "Notes"
		if folderName is "" then
			set f to default folder of default account
		else
			set f to folder folderName of default account
		end if
		set x to make new note at f with properties {{body:theBody}}
		return my clean(id of x) & tab & my clean(name of x) & tab & my clean(name of f)
	end tell
end run"""

ALL_SCRIPTS = (SEARCH, READ, CREATE)


def note_html(title: str, text: str) -> str:
    """The body Notes stores: the title as a heading, then one div per line."""
    lines = [f"<h1>{html.escape(title)}</h1>"]
    lines += [f"<div>{html.escape(line) or '<br>'}</div>" for line in text.splitlines()]
    return "".join(lines)


def _when(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text).astimezone() if text else None
    except ValueError:
        return None


class MacNotes(NotesApp):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._script = AppleScript("Notes", runner)

    def search(self, words: str, limit: int = 20) -> list[Note]:
        found = self._script(SEARCH, f"search your notes for “{words}”", words, str(limit))
        notes = [Note(i, t, f, _when(m), x) for i, t, f, m, x in records(found, 5)]
        oldest = datetime(1900, 1, 1).astimezone()
        return sorted(notes, key=lambda n: n.modified or oldest, reverse=True)

    def read(self, note_id: str) -> Note:
        found = self._script(READ, "read the note", note_id)
        head, _, text = found.partition("\n")
        if head.strip() == LOCKED:
            raise ToolExecutionError("That note is locked.", hint="Open it in Notes yourself.")
        rows = records(head, 4) or [["", "", "", ""]]
        found_id, title, folder, modified = rows[0]
        return Note(found_id, title, folder, _when(modified), text.rstrip("\n"))

    def create(self, title: str, text: str, folder: str | None = None) -> Note:
        body = note_html(title, text)
        reply = self._script(CREATE, f"create the note “{title}”", body, folder or "")
        rows = records(reply, 3) or [["", title, folder or ""]]
        note_id, saved_title, saved_folder = rows[0]
        return Note(note_id, saved_title or title, saved_folder, datetime.now().astimezone(), text)
