"""Ring timers and alarms when they come due.

The ringer is a thread in the background service. It sleeps until the next
timer is due or until the timers change (``changed()``), and never wakes up
while nothing is pending. With timers pending it wakes at least once a
minute: the monotonic clock stops while the Mac sleeps, so a timer that came
due during sleep rings within a minute of waking.

Each timer rings once, even with several processes using the database:
``mark_fired`` succeeds for one of them only. A timer that came due while the
service was stopped rings when it starts, saying when it was due.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from jarvis.timers.store import Timer, TimerStore, TimerStoreError

log = logging.getLogger(__name__)

MAX_WAIT_SECONDS = 60.0
LATE = timedelta(minutes=1)  # later than this, the message says when it was due

Ring = Callable[[Timer, timedelta], None]


def _now() -> datetime:
    return datetime.now(UTC)


def ring_message(timer: Timer, late_by: timedelta) -> str:
    """ "Your “tea” timer is done.", "Reminder: buy milk.", "It's 7:00 AM: your alarm." """
    if timer.kind == "timer":
        text = f"Your “{timer.label}” timer is done." if timer.label else "Your timer is done."
    elif timer.label:
        text = f"Reminder: {timer.label}."
    else:
        text = f"It's {timer.due_at.astimezone():%-I:%M %p}: your alarm."
    if late_by > LATE:
        text += (
            f" It was due at {timer.due_at.astimezone():%-I:%M %p}, while Jarvis wasn't running."
        )
    return text


class TimerRinger:
    def __init__(
        self,
        store: TimerStore,
        ring: Ring,
        *,
        clock: Callable[[], datetime] = _now,
        max_wait: float = MAX_WAIT_SECONDS,
    ) -> None:
        self._store = store
        self._ring = ring
        self._clock = clock
        self._max_wait = max_wait
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="timers", daemon=True)
        self._thread.start()

    def changed(self) -> bool:
        """The timers changed: re-read them now. False if the ringer isn't running."""
        self._wake.set()
        return self.running

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def ring_due(self) -> float | None:
        """Ring what's due; seconds until the next timer (None if nothing is pending)."""
        now = self._clock()
        for timer in self._store.due(now):
            if not self._store.mark_fired(timer.id):
                continue  # another process rang or cancelled it first
            try:
                self._ring(timer, now - timer.due_at)
            except Exception:
                log.exception("Couldn't ring timer %d", timer.id)
        next_due = self._store.next_due()
        return None if next_due is None else (next_due - self._clock()).total_seconds()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                wait = self.ring_due()
            except TimerStoreError as exc:
                log.error("Timers unavailable: %s", exc.message)
                wait = self._max_wait
            timeout = None if wait is None else min(max(wait, 0.0), self._max_wait)
            self._wake.wait(timeout)
