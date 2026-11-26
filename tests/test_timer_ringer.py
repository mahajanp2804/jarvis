from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.timers.ringer import TimerRinger, ring_message
from jarvis.timers.store import Timer, TimerStore

NOW = datetime(2026, 11, 26, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> TimerStore:
    return TimerStore(tmp_path / "timers.sqlite3", clock=clock)


def test_messages() -> None:
    due = datetime(2026, 11, 26, 7, 0).astimezone()
    tea = Timer(1, "timer", "tea", due, due)
    assert ring_message(tea, timedelta(seconds=2)) == "Your “tea” timer is done."
    assert ring_message(Timer(2, "timer", "", due, due), timedelta(0)) == "Your timer is done."
    assert ring_message(Timer(3, "alarm", "buy milk", due, due), timedelta(0)) == (
        "Reminder: buy milk."
    )
    assert ring_message(Timer(4, "alarm", "", due, due), timedelta(0)) == (
        "It's 7:00 AM: your alarm."
    )
    assert ring_message(tea, timedelta(hours=2)).endswith(
        "It was due at 7:00 AM, while Jarvis wasn't running."
    )


def test_rings_what_is_due_once(store: TimerStore, clock: Clock) -> None:
    rung: list[tuple[str, timedelta]] = []
    ringer = TimerRinger(store, lambda t, late: rung.append((t.label, late)), clock=clock)
    store.add("timer", NOW + timedelta(minutes=5), "tea")
    store.add("alarm", NOW + timedelta(hours=1), "standup")
    assert ringer.ring_due() == 300  # nothing due yet: wait 5 minutes
    clock.now = NOW + timedelta(minutes=5, seconds=3)
    assert ringer.ring_due() == pytest.approx(3600 - 303)
    assert rung == [("tea", timedelta(seconds=3))]
    assert ringer.ring_due() is not None and len(rung) == 1  # never twice
    other = TimerRinger(store, lambda t, late: rung.append(("other", late)), clock=clock)
    clock.now = NOW + timedelta(hours=2)
    ringer.ring_due()
    assert other.ring_due() is None  # the first process already rang it
    assert [label for label, _ in rung] == ["tea", "standup"]


def test_a_failing_ring_does_not_stop_the_others(store: TimerStore, clock: Clock) -> None:
    rung: list[str] = []

    def ring(timer: Timer, late: timedelta) -> None:
        if timer.label == "broken":
            raise RuntimeError("no sound device")
        rung.append(timer.label)

    store.add("timer", NOW + timedelta(seconds=1), "broken")
    store.add("timer", NOW + timedelta(seconds=2), "eggs")
    clock.now = NOW + timedelta(seconds=5)
    assert TimerRinger(store, ring, clock=clock).ring_due() is None
    assert rung == ["eggs"] and store.pending() == []


def test_the_thread_sleeps_until_changed(tmp_path: Path) -> None:
    """Real clock: no timers means no wake-ups; a new timer wakes the thread at once."""
    store = TimerStore(tmp_path / "timers.sqlite3")
    rung = threading.Event()
    ringer = TimerRinger(store, lambda t, late: rung.set(), max_wait=60)
    assert ringer.changed() is False  # not running yet
    ringer.start()
    try:
        assert ringer.running and ringer.changed() is True
        store.add("timer", datetime.now(UTC) + timedelta(milliseconds=200), "quick")
        started = time.monotonic()
        ringer.changed()
        assert rung.wait(5) and time.monotonic() - started < 3
    finally:
        ringer.stop()
    assert not ringer.running
