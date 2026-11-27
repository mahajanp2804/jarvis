"""Notes tools: search your notes, read one, create one.

* Searching and reading are read-only; what a note says is outside content
  (shown to the model as data, never as instructions).
* Creating is LOW risk: a new note changes nothing that exists. After the
  request has read outside content, it asks first.
* Reading picks the note by title: an exact title wins, and when several
  notes match, Jarvis asks which one.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.dates import describe_day
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.platforms.base import Note, NotesApp, PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec

MAX_LISTED = 10
MAX_READ = 4000


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    words: str = Field(min_length=1, max_length=80, description="words in the title or text")


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=80, description="the note's title")


class CreateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=100)
    text: str = Field(default="", max_length=20_000)
    folder: str | None = Field(default=None, max_length=100)


class NoteTools:
    """Builds the ``notes.*`` tool specs."""

    def __init__(
        self,
        platform: PlatformAdapter,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._platform = platform
        self._clock = clock

    def _app(self) -> NotesApp:
        return self._platform.notes()  # UnsupportedPlatformError off macOS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("notes.search", "Search your notes.", SearchArgs, RiskLevel.READ_ONLY,
                     self._prepare_search, idempotent=True),
            ToolSpec("notes.read", "Read a note by its title.", ReadArgs, RiskLevel.READ_ONLY,
                     self._prepare_read, idempotent=True),
            ToolSpec("notes.create", "Create a note.", CreateArgs, RiskLevel.LOW,
                     self._prepare_create),
        ]  # fmt: skip

    def _edited(self, note: Note) -> str:
        if note.modified is None:
            return ""
        return f", edited {describe_day(note.modified.astimezone().date(), self._clock().date())}"

    def _prepare_search(self, args: SearchArgs) -> PreparedAction:
        app = self._app()
        words = " ".join(args.words.split())

        def run(cancel: threading.Event) -> ToolResult:
            found = app.search(words, limit=MAX_LISTED + 1)
            if not found:
                message = f"No notes mention “{words}”."
            else:
                lines = []
                for index, note in enumerate(found[:MAX_LISTED], start=1):
                    excerpt = " ".join(note.text.split())
                    if len(excerpt) > 120:
                        excerpt = excerpt[:120] + "…"
                    suffix = f": {excerpt}" if excerpt else ""
                    where = f"{note.folder}{self._edited(note)}"
                    lines.append(f"  {index}. {note.title} ({where}){suffix}")
                more = "\n  … and more" if len(found) > MAX_LISTED else ""
                message = f"Notes mentioning “{words}”:\n" + "\n".join(lines) + more
            return ToolResult(
                tool="notes.search",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"titles": [n.title for n in found[:MAX_LISTED]]},
                untrusted=bool(found),
            )

        return PreparedAction(description=f"Search your notes for “{words}”", run=run)

    def _prepare_read(self, args: ReadArgs) -> PreparedAction:
        app = self._app()
        wanted = " ".join(args.title.split())

        def run(cancel: threading.Event) -> ToolResult:
            found = app.search(wanted, limit=20)
            exact = [n for n in found if n.title.casefold() == wanted.casefold()]
            titled = exact or [n for n in found if wanted.casefold() in n.title.casefold()]
            if not titled:
                raise ToolExecutionError(f"There's no note titled “{wanted}”.")
            if len(titled) > 1:
                options = ", ".join(
                    f"“{n.title}” ({n.folder}{self._edited(n)})" for n in titled[:5]
                )
                raise NeedsClarificationError(f"Which note: {options}?")
            note = app.read(titled[0].id)
            text = note.text
            if len(text) > MAX_READ:
                text = text[:MAX_READ] + "\n[… the rest of the note isn't shown]"
            return ToolResult(
                tool="notes.read",
                status=ToolStatus.SUCCESS,
                message=f"“{note.title}” ({note.folder}{self._edited(note)}):\n{text}",
                data={"title": note.title, "chars": len(note.text)},
                untrusted=True,
            )

        return PreparedAction(description=f"Read the note “{wanted}”", run=run)

    def _prepare_create(self, args: CreateArgs) -> PreparedAction:
        app = self._app()
        title = " ".join(args.title.split())
        text = args.text.strip()
        folder = (args.folder or "").strip() or None
        where = f" in the “{folder}” folder" if folder else ""

        def run(cancel: threading.Event) -> ToolResult:
            note = app.create(title, text, folder)
            return ToolResult(
                tool="notes.create",
                status=ToolStatus.SUCCESS,
                message=f"Created the note “{note.title}” in {note.folder or 'Notes'}.",
                data={"id": note.id},
            )

        return PreparedAction(description=f"Create the note “{title}”{where}", run=run)
