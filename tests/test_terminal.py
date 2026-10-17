"""terminal.* tools with real processes (Python scripts) in a temporary project."""

from __future__ import annotations

import os
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


@pytest.fixture
def project(home: Path) -> Path:
    path = home / "Desktop" / "demo"
    path.mkdir()
    scripts = {
        "ok.py": "print('hello from demo')",
        "broken.py": "import pandas_not_installed",
        "slow.py": "import time\ntime.sleep(30)",
        "env.py": "import os\nprint(sorted(os.environ))",
        "server.py": """
            import time
            print('Running on http://127.0.0.1:8765 (Press CTRL+C to quit)', flush=True)
            time.sleep(60)
        """,
        "crash_server.py": "raise SystemExit('config missing')",
    }
    for name, body in scripts.items():
        (path / name).write_text(textwrap.dedent(body).strip() + "\n")
    return path


@pytest.fixture
def terminal_runtime(
    make_runtime: MakeRuntime, fake_bin: Path, project: Path
) -> Callable[..., tuple[Runtime, ScriptedConfirmer]]:
    def factory(**kwargs: Any) -> tuple[Runtime, ScriptedConfirmer]:
        terminal = {
            "search_path": [str(fake_bin)],
            "startup_seconds": 5,
            **kwargs.pop("terminal", {}),
        }
        return make_runtime(terminal=terminal, **kwargs)

    return factory


def run(
    runtime: Runtime, command: str, *, background: bool = False, tainted: bool = False
) -> ToolResult:
    call = ToolCall(
        tool="terminal.run",
        arguments={"command": command, "folder": "~/Desktop/demo", "background": background},
    )
    return runtime.dispatcher.execute(call, ExecutionContext(tainted=tainted))


def test_run_success_asks_first_and_returns_untrusted_output(terminal_runtime: Any) -> None:
    runtime, confirmer = terminal_runtime(confirm=True)
    result = run(runtime, "python3 ok.py")
    assert result.status is ToolStatus.SUCCESS and result.verified is True
    assert "hello from demo" in result.message
    assert result.untrusted is True
    assert "Run: `python3 ok.py` in ~/Desktop/demo" in confirmer.prompts[0]
    assert "runs project code" in confirmer.prompts[0]


def test_read_only_command_needs_no_confirmation(terminal_runtime: Any) -> None:
    runtime, confirmer = terminal_runtime(confirm=False)
    result = run(runtime, "python3 --version")
    assert result.status is ToolStatus.SUCCESS
    assert confirmer.prompts == []


def test_tests_run_without_asking_unless_tainted(terminal_runtime: Any) -> None:
    runtime, confirmer = terminal_runtime(confirm=False)
    assert run(runtime, "pytest -q").status is ToolStatus.SUCCESS  # LOW risk (stub prints)
    assert run(runtime, "pytest -q", tainted=True).status is ToolStatus.CANCELLED
    assert len(confirmer.prompts) == 1


def test_failure_comes_with_a_diagnosis(terminal_runtime: Any) -> None:
    runtime, _ = terminal_runtime(confirm=True)
    result = run(runtime, "python3 broken.py")
    assert result.status is ToolStatus.FAILED and result.data["exit_code"] == 1
    assert result.hint and "Missing Python package 'pandas_not_installed'" in result.hint
    assert result.attempts == 1  # never retried: running code isn't idempotent


def test_timeout_kills_the_process_group(terminal_runtime: Any) -> None:
    runtime, _ = terminal_runtime(confirm=True, terminal={"timeout_seconds": 0.5})
    started = time.monotonic()
    result = run(runtime, "python3 slow.py")
    assert time.monotonic() - started < 6
    assert result.status is ToolStatus.FAILED and "still running after" in result.message
    assert result.hint and "background" in result.hint


def test_cancel_stops_a_running_command(terminal_runtime: Any) -> None:
    cancel = threading.Event()
    runtime, _ = terminal_runtime(confirm=True, cancel_event=cancel)
    threading.Timer(0.5, cancel.set).start()
    started = time.monotonic()
    result = run(runtime, "python3 slow.py")
    assert time.monotonic() - started < 6
    assert result.status is ToolStatus.CANCELLED


def test_environment_is_minimal(terminal_runtime: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_SECRET_TOKEN", "leak-me")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leak-me")
    runtime, _ = terminal_runtime(confirm=True)
    result = run(runtime, "python3 env.py")
    assert "JARVIS_SECRET_TOKEN" not in result.message
    assert "AWS_SECRET_ACCESS_KEY" not in result.message
    assert "'PATH'" in result.message and "'HOME'" in result.message


def test_server_runs_in_background_and_is_managed(terminal_runtime: Any) -> None:
    runtime, confirmer = terminal_runtime(confirm=True)
    started = run(runtime, "python3 server.py", background=True)
    assert started.status is ToolStatus.SUCCESS and started.verified is True
    assert started.data["url"] == "http://127.0.0.1:8765"
    assert "Start in the background" in confirmer.prompts[0]
    pid = started.data["pid"]
    listed = runtime.dispatcher.execute(ToolCall(tool="terminal.list", arguments={}))
    assert "p1: `python3 server.py`" in listed.message and "running" in listed.message
    output = runtime.dispatcher.execute(ToolCall(tool="terminal.output", arguments={"id": "p1"}))
    assert "Running on http://127.0.0.1:8765" in output.message and output.untrusted
    stopped = runtime.dispatcher.execute(ToolCall(tool="terminal.stop", arguments={"id": "p1"}))
    assert stopped.status is ToolStatus.SUCCESS and stopped.verified
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_server_that_exits_immediately_is_a_failure(terminal_runtime: Any) -> None:
    runtime, _ = terminal_runtime(confirm=True)
    result = run(runtime, "python3 crash_server.py", background=True)
    assert result.status is ToolStatus.FAILED and "exited right away" in result.message
    assert "config missing" in result.message


def test_runtime_close_stops_background_processes(terminal_runtime: Any) -> None:
    runtime, _ = terminal_runtime(confirm=True)
    pid = run(runtime, "python3 server.py", background=True).data["pid"]
    runtime.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    ("command", "folder"),
    [("rm -rf demo", "~/Desktop/demo"), ("python3 ok.py", "/etc"), ("ls | wc", "~/Desktop/demo")],
)
def test_refusals(terminal_runtime: Any, command: str, folder: str) -> None:
    runtime, confirmer = terminal_runtime(confirm=True)
    result = runtime.dispatcher.execute(
        ToolCall(tool="terminal.run", arguments={"command": command, "folder": folder})
    )
    assert result.status is ToolStatus.DENIED
    assert confirmer.prompts == []


def test_open_terminal(terminal_runtime: Any, fake_platform: FakePlatform, project: Path) -> None:
    runtime, _ = terminal_runtime()
    result = runtime.dispatcher.execute(
        ToolCall(tool="terminal.open", arguments={"folder": "~/Desktop/demo"})
    )
    assert result.status is ToolStatus.SUCCESS
    assert fake_platform.opened == [(project, "Terminal")]
