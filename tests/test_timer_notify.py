from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from jarvis.api.schemas import ToolCall
from jarvis.bootstrap import Runtime
from jarvis.runtime.control import ControlServer
from jarvis.timers.notify import TimerNotifier
from tests.fakes import ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


@pytest.fixture
def short_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="jt", dir="/tmp") as path:  # short socket paths
        yield Path(path)


def test_the_service_uses_its_own_ringer() -> None:
    notifier = TimerNotifier(lambda: Path("/nonexistent.sock"))
    assert notifier.changed() is False  # nothing running
    notifier.attach(lambda: True)
    assert notifier.changed() is True
    notifier.attach(None)
    assert notifier.changed() is False


def test_other_processes_message_the_service(short_dir: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handle(request: dict[str, Any]) -> dict[str, Any]:
        seen.append(request)
        return {"ok": True}

    server = ControlServer(short_dir / "j.sock", handle)
    server.start()
    try:
        assert TimerNotifier(lambda: short_dir / "j.sock").changed() is True
    finally:
        server.stop()
    assert seen == [{"cmd": "timers-changed"}]


def test_timers_set_without_a_service_say_so(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    result = runtime.dispatcher.execute(
        ToolCall(tool="timers.start", arguments={"duration": "5 minutes", "label": "tea"})
    )
    assert result.ok and "won't ring until you start it" in result.message
    assert runtime.timers is not None and [t.label for t in runtime.timers.pending()] == ["tea"]
