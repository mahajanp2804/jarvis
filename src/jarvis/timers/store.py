"""The timers database: one row per timer or alarm.

* Its own file (``~/Library/Application Support/Jarvis/timers.sqlite3`` on
  macOS), not the memory database: timers work with memory turned off, and
  ``jarvis memory clear`` doesn't cancel an alarm. Owner-only (0600).
* Several processes may use it at once (the service rings timers that a
  ``jarvis`` command in a terminal set); SQLite's locking keeps that safe.
* A timer is *pending* until it rings (*fired*) or is *cancelled*. Ended
  timers are deleted after a day.
* Labels are redacted with the same rules as the logs.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from jarvis.errors import JarvisError, ToolInputError
from jarvis.logging_setup import redact

MAX_PENDING = 50
MAX_LABEL = 100
KEEP_ENDED = timedelta(days=1)

Kind = Literal["timer", "alarm"]

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS timers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        label TEXT NOT NULL,
        due_at TEXT NOT NULL,
        created_at TEXT NOT NULL,
        duration_seconds INTEGER,
        state TEXT NOT NULL DEFAULT 'pending',
        ended_at TEXT
    );
    CREATE INDEX IF NOT EXISTS timers_pending ON timers (state, due_at);
    PRAGMA user_version = 1;
"""


class TimerStoreError(JarvisError):
    """The timers database can't be used."""


@dataclass(frozen=True)
class Timer:
    id: int
    kind: Kind
    label: str
    due_at: datetime  # timezone-aware
    created_at: datetime
    duration_seconds: int | None = None  # timers only: "10 minutes"


def _now() -> datetime:
    return datetime.now(UTC)


class TimerStore:
    """Thread-safe access to the timers database. Opens the file on first use."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime] = _now) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.path.exists():
                os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
            conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                conn.close()
                raise TimerStoreError(
                    f"The timers database was written by a newer Jarvis (schema {version}).",
                    hint="Update Jarvis, or move the file away to start fresh.",
                )
            conn.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise TimerStoreError(
                f"Can't open the timers database {self.path}: {exc}",
                hint="Move the file away (or delete it) to start with no timers.",
            ) from exc
        self._conn = conn
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @staticmethod
    def _row(row: sqlite3.Row) -> Timer:
        return Timer(
            id=row["id"],
            kind=row["kind"],
            label=row["label"],
            due_at=datetime.fromisoformat(row["due_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            duration_seconds=row["duration_seconds"],
        )

    def add(
        self, kind: Kind, due_at: datetime, label: str = "", duration_seconds: int | None = None
    ) -> Timer:
        if due_at.tzinfo is None:
            raise ValueError("due_at must be timezone-aware")
        label = redact(" ".join(label.split()))[:MAX_LABEL]
        now = self._clock()
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "DELETE FROM timers WHERE state != 'pending' AND ended_at < ?",
                        ((now - KEEP_ENDED).isoformat(),),
                    )
                    count = conn.execute(
                        "SELECT COUNT(*) FROM timers WHERE state = 'pending'"
                    ).fetchone()[0]
                    if count >= MAX_PENDING:
                        raise ToolInputError(
                            f"You already have {count} timers and alarms.",
                            hint="Cancel some first.",
                        )
                    cursor = conn.execute(
                        "INSERT INTO timers (kind, label, due_at, created_at, duration_seconds)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (kind, label, due_at.astimezone(UTC).isoformat(), now.isoformat(),
                         duration_seconds),
                    )  # fmt: skip
            except sqlite3.Error as exc:
                raise TimerStoreError(f"Couldn't save the {kind}: {exc}") from exc
        return Timer(int(cursor.lastrowid or 0), kind, label, due_at, now, duration_seconds)

    def pending(self) -> list[Timer]:
        """Timers and alarms that haven't rung or been cancelled, soonest first."""
        with self._lock:
            try:
                rows = self._connect().execute(
                    "SELECT * FROM timers WHERE state = 'pending' ORDER BY due_at, id"
                )
                return [self._row(row) for row in rows]
            except sqlite3.Error as exc:
                raise TimerStoreError(f"Couldn't read the timers: {exc}") from exc

    def due(self, now: datetime | None = None) -> list[Timer]:
        moment = (now or self._clock()).astimezone(UTC)
        return [t for t in self.pending() if t.due_at <= moment]

    def next_due(self) -> datetime | None:
        pending = self.pending()
        return pending[0].due_at if pending else None

    def _end(self, timer_id: int, state: str) -> bool:
        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute(
                        "UPDATE timers SET state = ?, ended_at = ?"
                        " WHERE id = ? AND state = 'pending'",
                        (state, self._clock().isoformat(), timer_id),
                    )
            except sqlite3.Error as exc:
                raise TimerStoreError(f"Couldn't update the timer: {exc}") from exc
        return cursor.rowcount == 1

    def cancel(self, timer_id: int) -> bool:
        """False if it already rang, was cancelled, or never existed."""
        return self._end(timer_id, "cancelled")

    def mark_fired(self, timer_id: int) -> bool:
        """False if another process rang (or cancelled) it first."""
        return self._end(timer_id, "fired")
