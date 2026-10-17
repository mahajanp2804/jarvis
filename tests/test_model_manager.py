from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import ClassVar

import pytest

from jarvis.ai.model_manager import ModelManager
from jarvis.errors import JarvisError
from tests.fakes import FakeLLM


class ManualTimer:
    """Timer double that fires only when the test says so."""

    created: ClassVar[list[ManualTimer]] = []

    def __init__(self, interval: float, function: Callable[[], None]) -> None:
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False
        ManualTimer.created.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.function()


@pytest.fixture(autouse=True)
def _reset_timers() -> None:
    ManualTimer.created = []


def manager(llm: FakeLLM, idle: float = 60) -> ModelManager:
    return ModelManager(llm, idle, timer_factory=ManualTimer)


def test_not_loaded_until_first_use() -> None:
    llm = FakeLLM()
    mgr = manager(llm)
    assert not mgr.is_loaded
    assert llm.loads == 0
    with mgr.session() as backend:
        assert backend is llm
        assert llm.loaded
    assert mgr.stats.loads == 1
    assert mgr.stats.last_load_seconds is not None


def test_consecutive_requests_reuse_the_loaded_model() -> None:
    llm = FakeLLM()
    mgr = manager(llm)
    for _ in range(3):
        with mgr.session():
            pass
    assert llm.loads == 1
    assert mgr.stats.requests == 3
    # every use re-arms the idle timer; earlier timers are cancelled
    assert [t.cancelled for t in ManualTimer.created] == [True, True, False]
    assert all(t.daemon and t.started for t in ManualTimer.created)


def test_idle_timer_unloads() -> None:
    llm = FakeLLM()
    mgr = manager(llm, idle=120)
    with mgr.session():
        pass
    timer = ManualTimer.created[-1]
    assert timer.interval == 120
    timer.fire()
    assert not llm.loaded
    assert mgr.stats.unloads == 1
    with mgr.session():
        pass
    assert llm.loads == 2  # reloaded on demand


def test_superseded_timer_does_not_unload() -> None:
    llm = FakeLLM()
    mgr = manager(llm)
    with mgr.session():
        pass
    stale = ManualTimer.created[-1]
    with mgr.session():
        pass
    stale.fire()  # a timer that was already replaced must be ignored
    assert llm.loaded


def test_timer_cannot_unload_during_a_request() -> None:
    llm = FakeLLM()
    mgr = manager(llm)
    with mgr.session():
        pass
    timer = ManualTimer.created[-1]
    fired = threading.Event()

    with mgr.session():
        # Fire from another thread while the session holds the lock.
        thread = threading.Thread(target=lambda: (timer.fire(), fired.set()))
        thread.start()
        time.sleep(0.05)
        assert not fired.is_set(), "idle unload must wait for the active request"
        assert llm.loaded
    thread.join(timeout=2)
    assert fired.is_set()
    assert llm.loaded  # and the stale timer was invalidated by the newer session


def test_zero_timeout_unloads_after_each_request() -> None:
    llm = FakeLLM()
    mgr = manager(llm, idle=0)
    with mgr.session():
        assert llm.loaded
    assert not llm.loaded
    assert ManualTimer.created == []


def test_unload_now_and_shutdown() -> None:
    llm = FakeLLM()
    mgr = manager(llm)
    with mgr.session():
        pass
    mgr.shutdown()
    assert not llm.loaded
    assert ManualTimer.created[-1].cancelled
    mgr.shutdown()  # idempotent
    assert llm.unloads == 1


def test_load_failure_propagates_and_leaves_model_unloaded() -> None:
    llm = FakeLLM(load_error=JarvisError("no model", hint="download it"))
    mgr = manager(llm)
    with pytest.raises(JarvisError, match="no model"), mgr.session():
        pass
    assert not llm.loaded
    assert mgr.stats.loads == 0


def test_real_threading_timer_unloads() -> None:
    llm = FakeLLM()
    mgr = ModelManager(llm, idle_unload_seconds=0.05)
    with mgr.session():
        pass
    deadline = time.monotonic() + 2
    while llm.loaded and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not llm.loaded
