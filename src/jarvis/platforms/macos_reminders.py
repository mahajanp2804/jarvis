"""The Reminders app on macOS, through fixed AppleScripts.

Titles, list names and notes are passed as ``argv``; dates are passed as
numbers and built inside the script. Reminders syncs through iCloud, so a
reminder added here also alerts on your iPhone.
"""

from __future__ import annotations

import subprocess
from datetime import datetime

from jarvis.platforms.applescript import CLEAN, AppleScript, Runner, records
from jarvis.platforms.base import Reminder, RemindersApp

# AppleScript date helpers: ISO text out, and a date from numbers in.
DATES = """\
on pad(n)
	return text -2 thru -1 of ("0" & (n as text))
end pad
on iso(d)
	if d is missing value then return ""
	set s to time of d
	set dayPart to ((year of d) as text) & "-" & my pad((month of d) as integer)
	set dayPart to dayPart & "-" & my pad(day of d)
	return dayPart & "T" & my pad(s div 3600) & ":" & my pad((s mod 3600) div 60)
end iso
on mkdate(y, m, d, hh, mm)
	set t to current date
	set day of t to 1
	set year of t to y
	set month of t to m
	set day of t to d
	set time of t to (hh * 3600 + mm * 60)
	return t
end mkdate"""

LISTS = """\
tell application "Reminders"
	set out to ""
	repeat with n in (get name of every list)
		set out to out & n & linefeed
	end repeat
	return out
end tell"""

OPEN_REMINDERS = f"""\
{CLEAN}
{DATES}
on run argv
	set wanted to item 1 of argv
	set out to ""
	tell application "Reminders"
		if wanted is "" then
			set theLists to every list
		else
			set theLists to {{list wanted}}
		end if
		repeat with l in theLists
			set lname to my clean(name of l)
			repeat with r in (every reminder of l whose completed is false)
				set isAllDay to (allday due date of r is not missing value) as text
				set row to my clean(id of r) & tab & my clean(name of r) & tab & lname
				set out to out & row & tab & my iso(due date of r) & tab & isAllDay & linefeed
			end repeat
		end repeat
	end tell
	return out
end run"""

ADD = f"""\
{CLEAN}
{DATES}
on run argv
	set {{theTitle, listName, theNotes, hasDue, allDay}} to items 1 thru 5 of argv
	tell application "Reminders"
		if listName is "" then
			set l to default list
		else
			set l to list listName
		end if
		tell l to set r to make new reminder with properties {{name:theTitle}}
		if theNotes is not "" then set body of r to theNotes
		if hasDue is "1" then
			set n to {{}}
			repeat with i from 6 to 10
				set end of n to (item i of argv) as integer
			end repeat
			set dt to my mkdate(item 1 of n, item 2 of n, item 3 of n, item 4 of n, item 5 of n)
			if allDay is "1" then
				set allday due date of r to dt
			else
				set due date of r to dt
				set remind me date of r to dt
			end if
		end if
		return my clean(id of r) & tab & my clean(name of l)
	end tell
end run"""

COMPLETE = """\
on run argv
	tell application "Reminders"
		set completed of (reminder id (item 1 of argv)) to true
	end tell
end run"""

ALL_SCRIPTS = (LISTS, OPEN_REMINDERS, ADD, COMPLETE)


def _parse_due(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text).astimezone() if text else None
    except ValueError:
        return None


class MacReminders(RemindersApp):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._script = AppleScript("Reminders", runner)

    def lists(self) -> list[str]:
        listing = self._script(LISTS, "list your reminder lists")
        return [name.strip() for name in listing.splitlines() if name.strip()]

    def open_reminders(self, list_name: str | None = None) -> list[Reminder]:
        found = self._script(OPEN_REMINDERS, "read your reminders", list_name or "")
        return [
            Reminder(rid, title, lst, _parse_due(due), all_day == "true")
            for rid, title, lst, due, all_day in records(found, 5)
        ]

    def add(
        self,
        title: str,
        list_name: str | None = None,
        due: datetime | None = None,
        *,
        all_day: bool = False,
        notes: str = "",
    ) -> Reminder:
        args = [title, list_name or "", notes, "1" if due else "0", "1" if all_day else "0"]
        if due is not None:
            local = due.astimezone()
            args += [str(v) for v in (local.year, local.month, local.day, local.hour, local.minute)]
        reply = self._script(ADD, f"add the reminder “{title}”", *args)
        rows = records(reply, 2)
        rid, saved_list = rows[0] if rows else ("", list_name or "")
        return Reminder(rid, title, saved_list, due, all_day)

    def complete(self, reminder_id: str) -> None:
        self._script(COMPLETE, "complete the reminder", reminder_id)
