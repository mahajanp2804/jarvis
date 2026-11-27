"""The Calendar app on macOS, through fixed AppleScripts.

* Reading: for each calendar, the events overlapping the window plus the
  repeating events that started before it. AppleScript returns a repeating
  event once, at its first date, so its dates in the window are worked out
  from its rule (:mod:`jarvis.recurrence`). A rule that can't be expanded, or
  a calendar that won't list its repeating events, is reported in the
  listing's notes instead of being silently left out.
* Adding: titles and places go in ``argv``; dates go in as numbers.
* Edited single occurrences of a repeating event aren't read separately: the
  listing shows the occurrence at its usual time.
"""

from __future__ import annotations

import subprocess
from datetime import datetime

from jarvis.platforms.applescript import CLEAN, AppleScript, Runner, records
from jarvis.platforms.base import CalendarApp, CalendarEvent, EventListing
from jarvis.platforms.macos_reminders import DATES
from jarvis.recurrence import UnsupportedRule, occurrences

NO_REPEATS = "#norepeat"

CALENDARS = """\
tell application "Calendar"
	set out to ""
	repeat with c in (every calendar whose writable is true)
		set out to out & name of c & linefeed
	end repeat
	return out
end tell"""

EVENTS = f"""\
{CLEAN}
{DATES}
on row(e, cname)
	tell application "Calendar"
		set ex to ""
		try
			repeat with x in (excluded dates of e)
				set ex to ex & my iso(x) & ","
			end repeat
		end try
		set headPart to my clean(uid of e) & tab & my clean(summary of e) & tab & cname
		set timePart to my iso(start date of e) & tab & my iso(end date of e)
		set timePart to timePart & tab & ((allday event of e) as text)
		set tailPart to my clean(location of e) & tab & my clean(recurrence of e) & tab & ex
		return headPart & tab & timePart & tab & tailPart
	end tell
end row
on run argv
	set n to {{}}
	repeat with i from 1 to 10
		set end of n to (item i of argv) as integer
	end repeat
	set ws to my mkdate(item 1 of n, item 2 of n, item 3 of n, item 4 of n, item 5 of n)
	set we to my mkdate(item 6 of n, item 7 of n, item 8 of n, item 9 of n, item 10 of n)
	set out to ""
	tell application "Calendar"
		repeat with c in every calendar
			set cname to my clean(name of c)
			set found to {{}}
			try
				set found to (every event of c whose start date < we and end date > ws)
			end try
			set repeating to {{}}
			try
				set repeating to (every event of c whose start date < ws ¬
					and recurrence is not missing value)
			on error
				set out to out & "{NO_REPEATS}" & tab & cname & linefeed
			end try
			repeat with e in (found & repeating)
				set out to out & my row(e, cname) & linefeed
			end repeat
		end repeat
	end tell
	return out
end run"""

ADD_EVENT = f"""\
{CLEAN}
{DATES}
on run argv
	set {{theTitle, calName, theLocation, allDay}} to items 1 thru 4 of argv
	set n to {{}}
	repeat with i from 5 to 14
		set end of n to (item i of argv) as integer
	end repeat
	set s to my mkdate(item 1 of n, item 2 of n, item 3 of n, item 4 of n, item 5 of n)
	set e to my mkdate(item 6 of n, item 7 of n, item 8 of n, item 9 of n, item 10 of n)
	tell application "Calendar"
		if calName is "" then
			set c to first calendar whose writable is true
		else
			set c to calendar calName
		end if
		set props to {{summary:theTitle, start date:s, end date:e, allday event:(allDay is "1")}}
		tell c to set ev to make new event at end of events with properties props
		if theLocation is not "" then set location of ev to theLocation
		return my clean(uid of ev) & tab & my clean(name of c)
	end tell
end run"""

ALL_SCRIPTS = (CALENDARS, EVENTS, ADD_EVENT)


def _time(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text).astimezone() if text else None
    except ValueError:
        return None


def _numbers(moment: datetime) -> list[str]:
    local = moment.astimezone()
    return [str(v) for v in (local.year, local.month, local.day, local.hour, local.minute)]


def parse_events(text: str, start: datetime, end: datetime) -> EventListing:
    """Rows from the EVENTS script -> occurrences in [start, end)."""
    events: dict[tuple[str, datetime], CalendarEvent] = {}
    notes: list[str] = []
    for row in records(text, 9):
        if row[0] == NO_REPEATS:
            notes.append(f"Repeating events in “{row[1]}” couldn't be read and may be missing.")
            continue
        uid, title, calendar, first_text, last_text, all_day, location, rule, excluded = row
        first, last = _time(first_text), _time(last_text)
        if first is None or last is None:
            continue
        length = last - first
        if rule:
            skip = frozenset(
                d.date() for d in (_time(x) for x in excluded.split(",") if x) if d is not None
            )
            try:
                starts = occurrences(rule, first, start - length, end, skip)
            except UnsupportedRule:
                notes.append(f"“{title}” repeats in a way Jarvis can't work out; check Calendar.")
                continue
            starts = [s for s in starts if s + length > start]
        else:
            starts = [first] if first < end and last > start else []
        for occurrence in starts:
            events[(uid, occurrence)] = CalendarEvent(
                uid, title, calendar, occurrence, occurrence + length, all_day == "true", location
            )
    listing = sorted(events.values(), key=lambda e: (not e.all_day, e.start, e.title.casefold()))
    return EventListing(listing, notes)


class MacCalendar(CalendarApp):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        # Reading many calendars can take a while.
        self._script = AppleScript("Calendar", runner, timeout=60.0)

    def calendars(self) -> list[str]:
        listing = self._script(CALENDARS, "list your calendars")
        return [name.strip() for name in listing.splitlines() if name.strip()]

    def events(self, start: datetime, end: datetime) -> EventListing:
        found = self._script(EVENTS, "read your calendar", *_numbers(start), *_numbers(end))
        return parse_events(found, start, end)

    def add_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        *,
        all_day: bool = False,
        calendar: str | None = None,
        location: str = "",
    ) -> CalendarEvent:
        args = [title, calendar or "", location, "1" if all_day else "0"]
        reply = self._script(
            ADD_EVENT, f"add “{title}” to your calendar", *args, *_numbers(start), *_numbers(end)
        )
        rows = records(reply, 2)
        uid, used = rows[0] if rows else ("", calendar or "")
        return CalendarEvent(uid, title, used, start, end, all_day, location)
