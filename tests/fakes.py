"""In-memory test doubles for the platform adapter and confirmation prompts."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.ai.base import ChatMessage, Completion, LLMBackend
from jarvis.platforms.base import (
    AppControlAdapter,
    AppInfo,
    CalendarApp,
    CalendarEvent,
    Contact,
    ContactsBook,
    DiagnosticCheck,
    EventListing,
    MailApp,
    Messenger,
    MusicPlayer,
    Note,
    NotesApp,
    NowPlaying,
    PlatformAdapter,
    Reminder,
    RemindersApp,
    SystemControls,
    SystemState,
    TextLine,
)


def make_app(name: str, bundle_id: str | None = None, *alt: str) -> AppInfo:
    return AppInfo(
        name=name,
        path=Path(f"/Applications/{name}.app"),
        bundle_id=bundle_id if bundle_id is not None else f"com.example.{name.replace(' ', '')}",
        alt_names=tuple(alt),
    )


DEFAULT_APPS = [
    make_app("Safari", "com.apple.Safari"),
    make_app("Notes", "com.apple.Notes"),
    make_app("Photos", "com.apple.Photos"),
    make_app("Photo Booth", "com.apple.PhotoBooth"),
    make_app("Visual Studio Code", "com.microsoft.VSCode", "Code"),
    make_app("Calculator", "com.apple.calculator"),
    make_app("TextEdit", "com.apple.TextEdit"),
]


class FakeAppControl(AppControlAdapter):
    """Records calls and simulates running state without touching the OS."""

    def __init__(self, apps: Sequence[AppInfo] = DEFAULT_APPS) -> None:
        self.apps = list(apps)
        self.running: set[str] = set()
        self.launched: list[str] = []
        self.quit_requests: list[str] = []
        self.discover_calls = 0
        # Behaviour switches for failure-path tests
        self.launch_starts_app = True
        self.quit_stops_app = True
        self.can_detect_running = True
        self.launch_error: Exception | None = None

    def default_search_paths(self) -> list[Path]:
        return [Path("/Applications")]

    def default_aliases(self) -> dict[str, str]:
        return {"vscode": "Visual Studio Code", "chrome": "Google Chrome"}

    def discover(self, search_paths: Sequence[Path]) -> list[AppInfo]:
        self.discover_calls += 1
        return list(self.apps)

    def launch(self, app: AppInfo) -> None:
        if self.launch_error:
            raise self.launch_error
        self.launched.append(app.name)
        if self.launch_starts_app:
            self.running.add(app.name)

    def is_running(self, app: AppInfo) -> bool | None:
        if not self.can_detect_running:
            return None
        return app.name in self.running

    def quit(self, app: AppInfo) -> None:
        self.quit_requests.append(app.name)
        if self.quit_stops_app:
            self.running.discard(app.name)


class FakeSystemControls(SystemControls):
    """Volume, Wi-Fi and display in memory; records every change."""

    def __init__(self) -> None:
        self.current = SystemState(
            volume=50, muted=False, wifi_on=True, battery_percent=80,
            charging=False, on_battery_power=True, dark_mode=False,
        )  # fmt: skip
        self.changes: list[tuple[str, object]] = []

    def state(self) -> SystemState:
        return self.current

    def _set(self, what: str, value: object, **fields: object) -> None:
        from dataclasses import replace

        self.changes.append((what, value))
        self.current = replace(self.current, **fields)  # type: ignore[arg-type]

    def set_volume(self, level: int) -> None:
        self._set("volume", level, volume=level)

    def set_muted(self, muted: bool) -> None:
        self._set("muted", muted, muted=muted)

    def set_wifi(self, on: bool) -> None:
        self._set("wifi", on, wifi_on=on)

    def sleep_display(self) -> None:
        self.changes.append(("sleep_display", None))


class FakeMusicPlayer(MusicPlayer):
    """A tiny library in memory; records every command."""

    library: tuple[tuple[str, str, str], ...] = (
        ("So What", "Miles Davis", "Kind of Blue"),
        ("Take Five", "The Dave Brubeck Quartet", "Time Out"),
    )

    def __init__(self) -> None:
        self.current = NowPlaying("not running")
        self.playlist_names = ["Library", "Road Trip", "Focus"]
        self.commands: list[str] = []

    def now_playing(self) -> NowPlaying:
        return self.current

    def command(self, command: str) -> None:
        from dataclasses import replace

        self.commands.append(command)
        if command == "play":
            track, artist, album = self.library[0]
            self.current = replace(self.current, state="playing", track=track,
                                   artist=artist, album=album)  # fmt: skip
        elif command == "pause":
            self.current = replace(self.current, state="paused")
        else:
            track, artist, album = self.library[1]
            self.current = NowPlaying("playing", track, artist, album)

    def playlists(self) -> list[str]:
        if self.current.state == "not running":
            self.current = NowPlaying("stopped")  # listing playlists starts Music
        return list(self.playlist_names)

    def play_playlist(self, name: str) -> None:
        self.commands.append(f"playlist:{name}")
        self.current = NowPlaying("playing", *self.library[1])

    def play_search(self, query: str) -> NowPlaying | None:
        self.commands.append(f"search:{query}")
        for track, artist, album in self.library:
            if query.casefold() in f"{track} {artist} {album}".casefold():
                self.current = NowPlaying("playing", track, artist, album)
                return self.current
        return None


class FakeReminders(RemindersApp):
    """Reminder lists in memory."""

    def __init__(self) -> None:
        self.list_names = ["Reminders", "Groceries"]
        self.items: list[Reminder] = []
        self.completed: list[str] = []

    def lists(self) -> list[str]:
        return list(self.list_names)

    def open_reminders(self, list_name: str | None = None) -> list[Reminder]:
        return [
            r for r in self.items
            if r.id not in self.completed and (list_name is None or r.list_name == list_name)
        ]  # fmt: skip

    def add(
        self,
        title: str,
        list_name: str | None = None,
        due: datetime | None = None,
        *,
        all_day: bool = False,
        notes: str = "",
    ) -> Reminder:
        reminder = Reminder(
            f"r{len(self.items) + 1}", title, list_name or "Reminders", due, all_day
        )
        self.items.append(reminder)
        return reminder

    def complete(self, reminder_id: str) -> None:
        self.completed.append(reminder_id)


class FakeCalendar(CalendarApp):
    """Calendars and events in memory."""

    def __init__(self) -> None:
        self.calendar_names = ["Home", "Work"]
        self.items: list[CalendarEvent] = []
        self.notes: list[str] = []
        self.windows: list[tuple[datetime, datetime]] = []

    def calendars(self) -> list[str]:
        return list(self.calendar_names)

    def events(self, start: datetime, end: datetime) -> EventListing:
        self.windows.append((start, end))
        found = [e for e in self.items if e.start < end and e.end > start]
        return EventListing(sorted(found, key=lambda e: (not e.all_day, e.start)), self.notes)

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
        event = CalendarEvent(
            f"e{len(self.items) + 1}", title, calendar or "Home", start, end, all_day, location
        )
        self.items.append(event)
        return event


class FakeContacts(ContactsBook):
    """An address book in memory; your card lists Jane as your mother."""

    def __init__(self) -> None:
        self.people = [
            Contact("p1", "Jane Doe", (("mobile", "+1 555 0100"), ("home", "+1 555 0199")),
                    (("home", "jane@example.com"),)),
            Contact("p2", "Sam Lee", (("work", "+1 555 0142"),), (("work", "sam@example.org"),)),
            Contact("p3", "Sam Patel", (("mobile", "+1 555 0177"),), ()),
        ]  # fmt: skip
        self.relations = {"mother": "Jane Doe"}
        self.queries: list[tuple[str, str | None]] = []

    def find(self, name: str, relation: str | None = None) -> list[Contact]:
        self.queries.append((name, relation))
        found = [p for p in self.people if name.casefold() in p.name.casefold()]
        related = self.relations.get(relation or "")
        found += [p for p in self.people if p.name == related and p not in found]
        return found


class FakeNotes(NotesApp):
    """Notes in memory."""

    def __init__(self) -> None:
        self.items: list[Note] = []

    def search(self, words: str, limit: int = 20) -> list[Note]:
        key = words.casefold()
        found = [n for n in self.items if key in n.title.casefold() or key in n.text.casefold()]
        return [Note(n.id, n.title, n.folder, n.modified, n.text[:160]) for n in found[:limit]]

    def read(self, note_id: str) -> Note:
        return next(n for n in self.items if n.id == note_id)

    def create(self, title: str, text: str, folder: str | None = None) -> Note:
        note = Note(f"n{len(self.items) + 1}", title, folder or "Notes", None, f"{title}\n{text}")
        self.items.append(note)
        return note


class FakeMessenger(Messenger, MailApp):
    """Records what would have been sent; nothing leaves the test."""

    def __init__(self) -> None:
        self.texts: list[tuple[str, str]] = []
        self.emails: list[tuple[str, str, str, bool]] = []

    def send(self, handle: str, text: str) -> str:
        self.texts.append((handle, text))
        return "iMessage"

    def compose(self, to: str, subject: str, body: str, *, send: bool) -> None:
        self.emails.append((to, subject, body, send))


class FakePlatform(PlatformAdapter):
    name = "Fake"

    def __init__(self, apps: FakeAppControl | None = None) -> None:
        self._apps = apps or FakeAppControl()
        self.opened: list[tuple[Path, str | None]] = []
        self.trash: list[Path] = []
        self.index: list[Path] | None = None  # None = no search index (folder scan)
        self.captures = []
        self.shortcut_runs = []
        self.system = FakeSystemControls()
        self.player = FakeMusicPlayer()
        self.alerts: list[str] = []
        self.reminder_app = FakeReminders()
        self.calendar_app = FakeCalendar()
        self.contact_book = FakeContacts()
        self.notes_app = FakeNotes()
        self.outbox = FakeMessenger()

    def find_by_name(
        self, words: Sequence[str], roots: Sequence[Path], *, folders_only: bool
    ) -> list[Path] | None:
        return None if self.index is None else list(self.index)

    def open_path(self, path: Path, app: AppInfo | None = None) -> None:
        self.opened.append((path, app.name if app else None))

    def list_shortcuts(self) -> list[str]:
        return list(self.shortcut_names)

    def run_shortcut(
        self, name: str, input_text: str | None, *, timeout: float, cancel: threading.Event
    ) -> str:
        self.shortcut_runs.append((name, input_text))
        return self.shortcut_output

    def system_controls(self) -> SystemControls:
        return self.system

    def music(self) -> MusicPlayer:
        return self.player

    def play_alert(self, sound: str, repeats: int = 3) -> None:
        self.alerts.append(sound)

    def reminders(self) -> RemindersApp:
        return self.reminder_app

    def calendar(self) -> CalendarApp:
        return self.calendar_app

    def contacts(self) -> ContactsBook:
        return self.contact_book

    def notes(self) -> NotesApp:
        return self.notes_app

    def messages(self) -> Messenger:
        return self.outbox

    def mail(self) -> MailApp:
        return self.outbox

    shortcut_names: tuple[str, ...] = ("Water Eject", "Extract Audio", "Set Brightness")
    shortcut_output: str = ""
    shortcut_runs: list[tuple[str, str | None]]

    screen_allowed: bool = True
    screen_text: tuple[str, ...] = ("Build passed", "3 warnings")
    captures: list[Path]

    def screen_capture_allowed(self) -> bool | None:
        return self.screen_allowed

    def capture_screen(self, dest: Path) -> None:
        from jarvis.errors import OSPermissionError

        if not self.screen_allowed:
            raise OSPermissionError(
                "macOS doesn't allow Jarvis to capture the screen.", hint="grant it"
            )
        self.captures.append(dest)
        dest.write_bytes(b"\x89PNG fake")

    def recognize_text(self, image: Path) -> list[TextLine]:
        assert image.exists(), "OCR must run before the capture is deleted"
        return [TextLine(t, 0.99, 0.1, 0.1 * i, 0.5, 0.05) for i, t in enumerate(self.screen_text)]

    def open_terminal(self, folder: Path) -> None:
        self.opened.append((folder, "Terminal"))

    def move_to_trash(self, path: Path) -> Path | None:
        import shutil

        self.trash.append(path)
        bin_dir = Path(os.environ["HOME"]) / ".Trash"
        bin_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(bin_dir / path.name))
        return bin_dir / path.name

    @property
    def apps(self) -> FakeAppControl:
        return self._apps

    def diagnostics(self) -> list[DiagnosticCheck]:
        return [DiagnosticCheck("Fake platform", True, "ok")]


class ScriptedConfirmer:
    """Answers confirmations from a fixed value and records the prompts."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def confirm(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answer


def plan_json(
    action: str, message: str, *steps: tuple[str, dict[str, Any]], finish: bool = True
) -> str:
    """JSON for a plan, as the model would produce it."""
    return json.dumps(
        {
            "action": action,
            "finish": finish,
            "message": message,
            "steps": [{"tool": tool, "arguments": args} for tool, args in steps],
        }
    )


class FakeLLM(LLMBackend):
    """Scripted model: returns queued replies and records every prompt."""

    name = "fake"

    def __init__(self, *replies: str, load_error: Exception | None = None) -> None:
        self.replies: deque[str] = deque(replies)
        self.load_error = load_error
        self.loaded = False
        self.loads = 0
        self.unloads = 0
        self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        if self.loaded:
            self.unloads += 1
            self.loaded = False

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if not self.loaded:
            raise AssertionError("complete_json called before load()")
        self.calls.append((list(messages), schema))
        text = (
            self.replies.popleft() if self.replies else plan_json("respond", "No scripted reply.")
        )
        return Completion(text=text, prompt_tokens=120, completion_tokens=24, seconds=0.01)
