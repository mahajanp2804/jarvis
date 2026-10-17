"""On-demand model lifecycle: load lazily, reuse while busy, unload when idle.

* The model is **not** loaded at start-up. The first request that needs it
  loads it.
* Consecutive requests reuse the loaded model (and llama.cpp's prompt cache).
* After ``idle_unload_seconds`` without use, a one-shot timer frees the model.
  There is no polling: an idle timer thread just sleeps until it fires.
* Access is serialised with a lock because a llama.cpp context is not
  thread-safe, and so the idle timer can never unload a model mid-request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from jarvis.ai.base import LLMBackend

log = logging.getLogger(__name__)


class TimerLike(Protocol):
    """The subset of :class:`threading.Timer` the manager uses (fakeable in tests)."""

    daemon: bool

    def start(self) -> None: ...
    def cancel(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], TimerLike]


@dataclass
class ModelStats:
    loads: int = 0
    unloads: int = 0
    last_load_seconds: float | None = None
    requests: int = 0


class ModelManager:
    """Owns a backend and decides when it occupies memory."""

    def __init__(
        self,
        backend: LLMBackend,
        idle_unload_seconds: float,
        *,
        timer_factory: TimerFactory = threading.Timer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.idle_unload_seconds = idle_unload_seconds
        self.stats = ModelStats()
        self._timer_factory = timer_factory
        self._clock = clock
        self._lock = threading.RLock()
        self._timer: TimerLike | None = None
        self._generation = 0  # invalidates timers that were superseded or cancelled

    @property
    def is_loaded(self) -> bool:
        return self.backend.is_loaded

    @contextmanager
    def session(self) -> Iterator[LLMBackend]:
        """Exclusive use of a loaded model. Loads on first use; arms the idle timer after."""
        with self._lock:
            self._cancel_timer()
            if not self.backend.is_loaded:
                started = self._clock()
                self.backend.load()
                self.stats.loads += 1
                self.stats.last_load_seconds = self._clock() - started
            try:
                self.stats.requests += 1
                yield self.backend
            finally:
                self._after_use()

    def abort(self) -> None:
        """Interrupt an in-flight request without waiting for the lock (shutdown path)."""
        self.backend.abort()

    def unload(self) -> None:
        """Free the model now (e.g. on shutdown or memory pressure)."""
        with self._lock:
            self._cancel_timer()
            self._unload_locked()

    shutdown = unload

    # -- internals -------------------------------------------------------------

    def _after_use(self) -> None:
        if self.idle_unload_seconds <= 0:
            self._unload_locked()
            return
        self._generation += 1
        generation = self._generation
        timer = self._timer_factory(self.idle_unload_seconds, lambda: self._on_idle(generation))
        timer.daemon = True  # never keeps the process alive
        self._timer = timer
        timer.start()

    def _on_idle(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return  # a newer request re-armed (or cancelled) the timer
            self._timer = None
            log.info("Model idle for %.0fs; unloading", self.idle_unload_seconds)
            self._unload_locked()

    def _cancel_timer(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _unload_locked(self) -> None:
        if self.backend.is_loaded:
            self.backend.unload()
            self.stats.unloads += 1
