from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from tests.fakes import FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def run(runtime: Runtime, tool: str, **arguments: Any) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool=tool, arguments=arguments))


@pytest.fixture
def tree(home: Path) -> Path:
    project = home / "Desktop" / "Code" / "stock-price-predictor"
    project.mkdir(parents=True)
    (project / "README.md").write_text("# Stock price predictor\n")
    (home / "Desktop" / "Code" / "weather-app").mkdir()
    (home / "Downloads" / "photo.JPG").write_bytes(b"\xff\xd8jpeg")
    (home / "Downloads" / "report.pdf").write_bytes(b"%PDF-1.4")
    (home / "Downloads" / "notes.txt").write_text("hello\nworld\n")
    (home / "Downloads" / ".hidden").write_text("x")
    (home / "Downloads" / "node_modules").mkdir()
    (home / "Library" / "secret.txt").write_text("nope")
    return home


def test_search_scans_and_ranks(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "files.search", query="stock price predictor", kind="folder")
    assert result.status is ToolStatus.SUCCESS
    assert result.data["results"][0]["path"] == "~/Desktop/Code/stock-price-predictor"
    assert result.data["source"] == "folder scan"
    assert not result.untrusted  # names only


def test_search_filters_index_results_through_the_policy(
    make_runtime: MakeRuntime, tree: Path, fake_platform: FakePlatform
) -> None:
    fake_platform.index = [
        tree / "Library" / "secret.txt",
        tree / "Downloads" / "notes.txt",
        Path("/etc/notes.txt"),
    ]
    runtime, _ = make_runtime()
    result = run(runtime, "files.search", query="notes")
    assert [r["path"] for r in result.data["results"]] == ["~/Downloads/notes.txt"]
    assert result.data["source"] == "Spotlight"


def test_list_hides_hidden_entries(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "files.list", folder="~/Downloads")
    names = [e["name"] for e in result.data["entries"]]
    assert names == ["node_modules", "notes.txt", "photo.JPG", "report.pdf"]
    assert ".hidden" not in result.message


def test_read_text_marks_content_untrusted(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "files.read", path="~/Downloads/notes.txt")
    assert result.status is ToolStatus.SUCCESS
    assert "hello\nworld" in result.message
    assert result.untrusted is True


@pytest.mark.parametrize(
    ("path", "status", "match"),
    [
        ("~/Downloads/photo.JPG", ToolStatus.FAILED, "not a text file"),
        ("~/Library/secret.txt", ToolStatus.DENIED, "protected"),
        ("~/Downloads/.hidden", ToolStatus.DENIED, "hidden"),
        ("~/Downloads", ToolStatus.FAILED, "not a file"),
        ("/etc/hosts", ToolStatus.DENIED, "outside"),
    ],
)
def test_read_refusals(
    make_runtime: MakeRuntime, tree: Path, path: str, status: ToolStatus, match: str
) -> None:
    runtime, _ = make_runtime()
    result = run(runtime, "files.read", path=path)
    assert result.status is status
    assert match in result.message


def test_read_truncates_and_enforces_size_limit(make_runtime: MakeRuntime, tree: Path) -> None:
    (tree / "Downloads" / "big.txt").write_text("x" * 5000)
    runtime, _ = make_runtime()
    result = run(runtime, "files.read", path="~/Downloads/big.txt", max_chars=200)
    assert "4800 more characters" in result.message
    strict, _ = make_runtime(files={"max_read_bytes": 1000})
    refused = run(strict, "files.read", path="~/Downloads/big.txt")
    assert refused.status is ToolStatus.FAILED and "reading limit" in refused.message


def test_create_never_overwrites(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime()
    created = run(runtime, "files.create", path="~/Desktop/todo.txt", content="buy milk")
    assert created.status is ToolStatus.SUCCESS and created.verified
    assert (tree / "Desktop" / "todo.txt").read_text() == "buy milk"
    again = run(runtime, "files.create", path="~/Desktop/todo.txt", content="other")
    assert again.status is ToolStatus.FAILED and "never overwrites" in again.message
    assert (tree / "Desktop" / "todo.txt").read_text() == "buy milk"
    outside = run(runtime, "files.create", path="~/Pictures/x.txt")
    assert outside.status is ToolStatus.DENIED


def test_create_folder_and_missing_parent(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime()
    assert run(runtime, "files.create", path="~/Desktop/New", kind="folder").verified
    missing = run(runtime, "files.create", path="~/Desktop/nope/x.txt")
    assert missing.status is ToolStatus.FAILED and "doesn't exist" in missing.message


def test_rename_and_move_ask_first_and_never_overwrite(
    make_runtime: MakeRuntime, tree: Path
) -> None:
    runtime, confirmer = make_runtime(confirm=True)
    renamed = run(runtime, "files.rename", path="~/Downloads/notes.txt", new_name="ideas.txt")
    assert renamed.status is ToolStatus.SUCCESS and (tree / "Downloads" / "ideas.txt").exists()
    assert "Rename ~/Downloads/notes.txt to ideas.txt" in confirmer.prompts[0]
    moved = run(runtime, "files.move", path="~/Downloads/ideas.txt", to_folder="~/Desktop")
    assert moved.status is ToolStatus.SUCCESS and (tree / "Desktop" / "ideas.txt").exists()
    (tree / "Downloads" / "ideas.txt").write_text("second")
    clash = run(runtime, "files.move", path="~/Downloads/ideas.txt", to_folder="~/Desktop")
    assert clash.status is ToolStatus.FAILED and "already exists" in clash.message
    bad_name = run(runtime, "files.rename", path="~/Downloads/ideas.txt", new_name="../x")
    assert bad_name.error_type == "ToolInputError"


def test_top_level_folders_cannot_be_moved(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, _ = make_runtime(confirm=True)
    result = run(runtime, "files.move", path="~/Downloads", to_folder="~/Desktop")
    assert result.status is ToolStatus.DENIED


def test_trash_is_high_risk_and_recoverable(
    make_runtime: MakeRuntime, tree: Path, fake_platform: FakePlatform
) -> None:
    runtime, confirmer = make_runtime(confirm=False, policy={"confirm_at_or_above": "high"})
    declined = run(runtime, "files.trash", path="~/Downloads/report.pdf")
    assert declined.status is ToolStatus.CANCELLED
    assert (tree / "Downloads" / "report.pdf").exists()
    assert "Trash" in confirmer.prompts[0]
    runtime2, _ = make_runtime(confirm=True)
    trashed = run(runtime2, "files.trash", path="~/Downloads/report.pdf")
    assert trashed.status is ToolStatus.SUCCESS and trashed.verified
    assert fake_platform.trash == [tree / "Downloads" / "report.pdf"]


def test_organize_one_confirmation_with_undo_list(
    make_runtime: MakeRuntime, tree: Path, tmp_path: Path
) -> None:
    (tree / "Downloads" / "Images").mkdir()
    (tree / "Downloads" / "Images" / "photo.JPG").write_bytes(b"older")  # collision -> skipped
    (tree / "Downloads" / "song.mp3").write_bytes(b"ID3")
    runtime, confirmer = make_runtime(confirm=True)
    result = run(runtime, "files.organize", folder="~/Downloads")
    assert len(confirmer.prompts) == 1
    prompt = confirmer.prompts[0]
    assert "move 3 file(s)" in prompt and "1 skipped" in prompt
    assert result.status is ToolStatus.SUCCESS
    downloads = tree / "Downloads"
    assert (downloads / "PDFs" / "report.pdf").exists()
    assert (downloads / "Documents" / "notes.txt").exists()
    assert (downloads / "Audio" / "song.mp3").exists()
    assert (downloads / "photo.JPG").exists()  # not moved: name taken in Images
    assert (downloads / "Images" / "photo.JPG").read_bytes() == b"older"
    undo = json.loads(Path(result.data["undo_list"]).read_text())
    assert len(undo["moves"]) == 3


def test_organize_nothing_to_do_is_a_noop(make_runtime: MakeRuntime, tree: Path) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = run(runtime, "files.organize", folder="~/Desktop")
    assert result.status is ToolStatus.SUCCESS and confirmer.prompts == []


def test_open_with_app(make_runtime: MakeRuntime, tree: Path, fake_platform: FakePlatform) -> None:
    runtime, _ = make_runtime()
    project = "~/Desktop/Code/stock-price-predictor"
    result = run(runtime, "files.open", path=project, app="vscode")
    assert result.status is ToolStatus.SUCCESS
    assert fake_platform.opened == [
        (tree / "Desktop" / "Code" / "stock-price-predictor", "Visual Studio Code")
    ]
