from __future__ import annotations

import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.errors import ToolInputError
from jarvis.timers.store import MAX_PENDING, TimerStore, TimerStoreError

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
    return TimerStore(tmp_path / "jarvis" / "timers.sqlite3", clock=clock)


def test_add_and_list_soonest_first(store: TimerStore) -> None:
    alarm = store.add("alarm", NOW + timedelta(hours=22), "wake up")
    tea = store.add("timer", NOW + timedelta(minutes=4), " tea  ", duration_seconds=240)
    assert [t.id for t in store.pending()] == [tea.id, alarm.id]
    assert store.pending()[0].label == "tea" and store.pending()[0].duration_seconds == 240
    assert store.next_due() == NOW + timedelta(minutes=4)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_due_fire_and_cancel(store: TimerStore, clock: Clock) -> None:
    tea = store.add("timer", NOW + timedelta(minutes=4), "tea")
    eggs = store.add("timer", NOW + timedelta(minutes=9), "eggs")
    assert store.due() == []
    clock.now = NOW + timedelta(minutes=5)
    assert [t.id for t in store.due()] == [tea.id]
    assert store.mark_fired(tea.id) is True
    assert store.mark_fired(tea.id) is False  # another process rang it first
    assert store.cancel(eggs.id) is True and store.cancel(eggs.id) is False
    assert store.pending() == [] and store.next_due() is None


def test_ended_timers_are_deleted_after_a_day(store: TimerStore, clock: Clock) -> None:
    old = store.add("timer", NOW + timedelta(minutes=1), "old")
    store.mark_fired(old.id)
    clock.now = NOW + timedelta(days=2)
    store.add("timer", clock.now + timedelta(minutes=1), "new")
    with store._connect() as conn:  # the fired row is gone
        assert conn.execute("SELECT COUNT(*) FROM timers").fetchone()[0] == 1


def test_another_process_sees_the_same_timers(store: TimerStore, clock: Clock) -> None:
    store.add("alarm", NOW + timedelta(hours=1), "standup")
    other = TimerStore(store.path, clock=clock)
    assert [t.label for t in other.pending()] == ["standup"]


def test_labels_are_redacted_and_limits_hold(store: TimerStore) -> None:
    timer = store.add("timer", NOW + timedelta(minutes=1), "password=hunter2 " + "x" * 300)
    assert "hunter2" not in timer.label and len(timer.label) <= 100
    for _ in range(MAX_PENDING - 1):
        store.add("timer", NOW + timedelta(minutes=1))
    with pytest.raises(ToolInputError, match="already have 50"):
        store.add("timer", NOW + timedelta(minutes=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        store.add("timer", datetime(2026, 11, 26, 9, 5))


def test_unreadable_files_are_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "timers.sqlite3"
    path.write_bytes(b"not a database" * 100)
    with pytest.raises(TimerStoreError, match="Can't open the timers database"):
        TimerStore(path).pending()
