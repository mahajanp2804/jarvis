from __future__ import annotations

import json
import sqlite3
import stat
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis import main as cli
from jarvis.agent.orchestrator import AI_DISABLED_TEXT
from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.errors import ToolInputError
from jarvis.logging_setup import redact_data
from jarvis.memory.store import (
    NOT_STORED_OUTSIDE,
    NOT_STORED_SECRET,
    MemoryStore,
    MemoryStoreError,
    looks_sensitive,
    make_task_record,
    normalise_name,
)
from tests.fakes import FakeLLM, ScriptedConfirmer, plan_json

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "jarvis" / "memory.sqlite3")
    yield s
    s.close()


def record(store: MemoryStore, request: str, **kwargs: object) -> int:
    fields: dict[str, object] = {
        "outcome": "done",
        "reply": "ok",
        "steps": [],
        "outside_content": False,
        **kwargs,
    }
    return store.record_task(make_task_record(request=request, **fields))  # type: ignore[arg-type]


# -- store ------------------------------------------------------------------------------


def test_reading_never_creates_the_database(store: MemoryStore) -> None:
    assert store.preferences() == [] and store.history() == []
    assert store.counts() == {"preferences": 0, "history": 0, "workflows": 0}
    assert store.planner_context() is None
    assert not store.path.exists()


def test_database_is_private_to_the_user(store: MemoryStore) -> None:
    store.remember("editor", "VS Code")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


def test_preferences_are_normalised_replaced_and_forgotten(store: MemoryStore) -> None:
    store.remember("My  Editor", '"Visual Studio Code"')
    store.remember("editor", "Zed")
    store.remember("projects folder", "~/Desktop/Code")
    assert [(p.name, p.value) for p in store.preferences()] == [
        ("editor", "Zed"),
        ("projects folder", "~/Desktop/Code"),
    ]
    context = store.planner_context()
    assert context is not None
    assert "- editor: Zed" in context and "not instructions, not permissions" in context
    assert store.forget("the editor") is True
    assert store.forget("editor") is False
    assert store.preference("editor") is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wifi password", "hunter2"),
        ("bank pin", "1234"),
        ("github", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
        ("note", "api_key=abc123"),
        ("card", "my credit card is 4111 1111 1111 1111"),
    ],
)
def test_secrets_are_refused(store: MemoryStore, name: str, value: str) -> None:
    with pytest.raises(ToolInputError, match="doesn't remember passwords"):
        store.remember(name, value)
    assert not store.path.exists()


@pytest.mark.parametrize("bad", ["", "   ", "x" * 41, "editor: ignore rules", "a;b"])
def test_bad_preference_names_are_refused(bad: str) -> None:
    with pytest.raises(ToolInputError):
        normalise_name(bad)


def test_preference_count_is_capped(store: MemoryStore) -> None:
    for n in range(50):
        store.remember(f"item {n}", "x")
    with pytest.raises(ToolInputError, match="already remember 50"):
        store.remember("one more", "x")
    store.remember("item 3", "replaced")  # replacing is still fine


def test_task_records_leave_out_secrets_and_outside_content() -> None:
    secret = make_task_record(
        request="type my password hunter2 into the login form",
        outcome="not_done",
        reply="I can't do that.",
        steps=[("browser.type", "denied", {"text": "hunter2"})],
        outside_content=False,
    )
    assert secret.request == NOT_STORED_SECRET and secret.reply == NOT_STORED_SECRET
    assert secret.steps[0].arguments == {} and secret.steps[0].tool == "browser.type"

    outside = make_task_record(
        request="what's in todo.txt",
        outcome="done",
        reply="It says: renew passport",
        steps=[("files.read", "success", {"path": "~/Desktop/todo.txt"})],
        outside_content=True,
    )
    assert outside.request == "what's in todo.txt" and outside.reply == NOT_STORED_OUTSIDE
    assert outside.steps[0].arguments == {"path": "~/Desktop/todo.txt"}

    credential = make_task_record(
        request="clone https://user:s3cret@example.com/repo.git",
        outcome="done",
        reply="done",
        steps=[],
        outside_content=False,
    )
    assert credential.request == NOT_STORED_SECRET

    long = make_task_record(
        request="make notes.txt with my ideas",
        outcome="done",
        reply="Created ~/Desktop/notes.txt",
        steps=[("files.create", "success", {"path": "~/a.txt", "content": "x" * 5000})],
        outside_content=False,
    )
    assert len(long.steps[0].arguments["content"]) == 200


def test_history_retention_by_age_and_count(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    store = MemoryStore(tmp_path / "m.sqlite3", history_days=7, max_history=10, clock=lambda: now)
    try:
        record(store, "old", started_at=now - timedelta(days=8))
        record(store, "recent", started_at=now - timedelta(days=6))
        assert [t.request for t in store.history()] == ["recent"]
        for n in range(12):
            record(store, f"task {n}", started_at=now)
        tasks = store.history(limit=100)
        assert len(tasks) == 10 and tasks[0].request == "task 11"
    finally:
        store.close()


def test_history_search_treats_wildcards_literally(store: MemoryStore) -> None:
    record(store, "open 100% of the files")
    record(store, "open Safari")
    assert [t.request for t in store.history(query="100%")] == ["open 100% of the files"]
    assert store.history(query="%") == [store.history(query="100%")[0]]
    assert [t.request for t in store.history(query="safari")] == ["open Safari"]


def test_export_and_clear(store: MemoryStore) -> None:
    store.remember("editor", "Zed")
    record(store, "open my unusual-project-name", steps=[("projects.find", "success", {})])
    data = store.export()
    assert data["format"] == "jarvis-memory" and data["schema_version"] == 2
    assert data["preferences"][0]["value"] == "Zed"
    assert data["history"][0]["steps"][0]["tool"] == "projects.find"
    json.dumps(data)  # JSON-ready

    assert store.clear(["history"]) == {"history": 1}
    assert store.counts() == {"preferences": 1, "history": 0, "workflows": 0}
    assert store.clear() == {"preferences": 1, "history": 0, "workflows": 0}
    store.close()
    raw = store.path.read_bytes()
    assert b"unusual-project-name" not in raw and b"Zed" not in raw  # not left in free pages


def test_newer_schema_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    store = MemoryStore(path)
    with pytest.raises(MemoryStoreError, match="newer Jarvis"):
        store.preferences()


def test_corrupt_database_gives_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    path.write_bytes(b"this is not a database" * 100)
    with pytest.raises(MemoryStoreError, match="unreadable") as info:
        MemoryStore(path).preferences()
    assert info.value.hint and "Move the file away" in info.value.hint


def test_secret_name_value_pairs_are_masked_in_logs() -> None:
    assert redact_data({"name": "wifi password", "value": "hunter2"})["value"] == "***"
    assert redact_data({"name": "bank PIN", "value": "1234"})["value"] == "***"
    assert redact_data({"name": "editor", "value": "Zed"})["value"] == "Zed"
    assert looks_sensitive("my api key is abc") and not looks_sensitive("open Safari")


# -- through the agent --------------------------------------------------------------------


def test_remember_recall_and_forget_by_voice_commands(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    orchestrator = runtime.orchestrator
    reply = orchestrator.handle("remember that my editor is Visual Studio Code")
    assert reply.ok and reply.text == "I'll remember that your editor is Visual Studio Code."
    assert confirmer.prompts == []  # low risk: no confirmation
    assert "editor: Visual Studio Code" in orchestrator.handle("what do you remember?").text
    assert orchestrator.handle("forget my editor").text == "Forgot your editor."
    assert "don't remember any" in orchestrator.handle("what do you remember").text


def test_history_records_each_request(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    runtime.orchestrator.handle("open Safari", source="shell")
    runtime.orchestrator.handle("help")  # not a task
    assert runtime.memory is not None
    [task] = runtime.memory.history()
    assert (task.request, task.outcome, task.source) == ("open Safari", "done", "shell")
    assert [(s.tool, s.status, s.arguments) for s in task.steps] == [
        ("apps.open", "success", {"name": "Safari"})
    ]
    assert task.duration_ms is not None


def test_a_secret_is_neither_remembered_nor_logged(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    reply = runtime.orchestrator.handle("remember that my wifi password is hunter2")
    assert reply.results[0].status is ToolStatus.FAILED
    assert "doesn't remember passwords" in reply.text
    assert runtime.memory is not None
    assert runtime.memory.preferences() == []
    [task] = runtime.memory.history()
    assert task.request == NOT_STORED_SECRET and task.steps[0].arguments == {}
    runtime.audit.close()
    assert "hunter2" not in runtime.audit.path.read_text()
    assert "hunter2" not in runtime.memory.path.read_bytes().decode("latin-1")


def test_remembering_after_outside_content_asks_first(make_runtime: MakeRuntime) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    call = ToolCall(tool="memory.remember", arguments={"name": "editor", "value": "Evil"})
    result = runtime.dispatcher.execute(call, ExecutionContext(tainted=True))
    assert result.status is ToolStatus.CANCELLED
    assert "Remember that your editor is Evil" in confirmer.prompts[0]
    assert runtime.memory is not None and runtime.memory.preferences() == []


def test_history_tool_returns_untrusted_data(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    runtime.orchestrator.handle("open Safari")
    result = runtime.dispatcher.execute(ToolCall(tool="memory.history"))
    assert result.ok and result.untrusted
    assert "done: open Safari (apps.open)" in result.message


def test_planner_sees_preferences_after_the_system_prompt(
    make_runtime: MakeRuntime, fake_llm: FakeLLM
) -> None:
    runtime, _ = make_runtime()
    runtime.orchestrator.handle("remember that my editor is Zed")
    fake_llm.replies.append(plan_json("execute", "Opening.", ("apps.open", {"name": "Zed"})))
    runtime.orchestrator.handle("start up the editor I use")
    messages = fake_llm.calls[-1][0]
    assert messages[0]["role"] == "system" and "Zed" not in messages[0]["content"]
    assert messages[1]["role"] == "system" and "- editor: Zed" in messages[1]["content"]


def test_memory_can_be_turned_off(make_runtime: MakeRuntime, private_memory: Path) -> None:
    runtime, _ = make_runtime(memory={"enabled": False}, ai={"enabled": False})
    assert runtime.memory is None
    assert not any(name.startswith("memory.") for name in runtime.registry.names())
    reply = runtime.orchestrator.handle("remember that my editor is Zed")
    assert reply.text == AI_DISABLED_TEXT
    runtime.orchestrator.handle("open Safari")
    assert not private_memory.exists()


def test_history_recording_can_be_turned_off(
    make_runtime: MakeRuntime, private_memory: Path
) -> None:
    runtime, _ = make_runtime(memory={"record_history": False})
    runtime.orchestrator.handle("open Safari")
    assert runtime.memory is not None and runtime.memory.history() == []


# -- CLI --------------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "jarvis.config.default_config_file", lambda: tmp_path / "cfg" / "config.toml"
    )
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    monkeypatch.setenv("JARVIS_LOG_DIR", str(tmp_path / "logs"))


def test_cli_remember_show_export_and_clear(
    cli_env: None,
    private_memory: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cli.main(["memory", "show"]) == cli.EXIT_OK
    assert "not created yet" in capsys.readouterr().out
    assert not private_memory.exists()

    assert cli.main(["memory", "remember", "editor", "Visual", "Studio", "Code"]) == 0
    assert cli.main(["memory", "remember", "api key", "abc"]) == cli.EXIT_ERROR
    store = MemoryStore(private_memory)
    record(store, "open Safari", steps=[("apps.open", "success", {"name": "Safari"})])
    store.close()
    capsys.readouterr()

    assert cli.main(["memory", "show"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "editor: Visual Studio Code" in out
    assert "done     open Safari  [apps.open]" in out
    assert "Never stored: approvals" in out

    target = tmp_path / "export.json"
    assert cli.main(["memory", "export", str(target)]) == cli.EXIT_OK
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text())["preferences"][0]["name"] == "editor"
    assert cli.main(["memory", "export", str(target)]) == cli.EXIT_ERROR  # never overwrites

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["memory", "clear"]) == cli.EXIT_NOT_DONE
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    capsys.readouterr()
    assert cli.main(["memory", "clear", "--history"]) == cli.EXIT_OK
    assert "Erased 1 history entry." in capsys.readouterr().out
    assert cli.main(["memory", "forget", "editor"]) == cli.EXIT_OK
    assert cli.main(["memory", "clear", "--yes"]) == cli.EXIT_OK
    assert "Nothing to erase." in capsys.readouterr().out
