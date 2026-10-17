"""In-memory test doubles for the platform adapter and confirmation prompts."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jarvis.ai.base import ChatMessage, Completion, LLMBackend
from jarvis.platforms.base import AppControlAdapter, AppInfo, DiagnosticCheck, PlatformAdapter


def make_app(name: str, bundle_id: str | None = None, *alt: str) -> AppInfo:
    return AppInfo(
        name=name,
        path=Path(f"/Applications/{name}.app"),
        bundle_id=bundle_id if bundle_id is not None else f"com.example.{name.replace(' ', '')}",
        alt_names=tuple(alt),
    )


DEFAULT_APPS = [
    make_app("Safari", "com.apple.Safari"),
    make_app("Notes", "com.apple.Notes"),
    make_app("Photos", "com.apple.Photos"),
    make_app("Photo Booth", "com.apple.PhotoBooth"),
    make_app("Visual Studio Code", "com.microsoft.VSCode", "Code"),
    make_app("Calculator", "com.apple.calculator"),
    make_app("TextEdit", "com.apple.TextEdit"),
]


class FakeAppControl(AppControlAdapter):
    """Records calls and simulates running state without touching the OS."""

    def __init__(self, apps: Sequence[AppInfo] = DEFAULT_APPS) -> None:
        self.apps = list(apps)
        self.running: set[str] = set()
        self.launched: list[str] = []
        self.quit_requests: list[str] = []
        self.discover_calls = 0
        # Behaviour switches for failure-path tests
        self.launch_starts_app = True
        self.quit_stops_app = True
        self.can_detect_running = True
        self.launch_error: Exception | None = None

    def default_search_paths(self) -> list[Path]:
        return [Path("/Applications")]

    def default_aliases(self) -> dict[str, str]:
        return {"vscode": "Visual Studio Code", "chrome": "Google Chrome"}

    def discover(self, search_paths: Sequence[Path]) -> list[AppInfo]:
        self.discover_calls += 1
        return list(self.apps)

    def launch(self, app: AppInfo) -> None:
        if self.launch_error:
            raise self.launch_error
        self.launched.append(app.name)
        if self.launch_starts_app:
            self.running.add(app.name)

    def is_running(self, app: AppInfo) -> bool | None:
        if not self.can_detect_running:
            return None
        return app.name in self.running

    def quit(self, app: AppInfo) -> None:
        self.quit_requests.append(app.name)
        if self.quit_stops_app:
            self.running.discard(app.name)


class FakePlatform(PlatformAdapter):
    name = "Fake"

    def __init__(self, apps: FakeAppControl | None = None) -> None:
        self._apps = apps or FakeAppControl()
        self.opened: list[tuple[Path, str | None]] = []
        self.trash: list[Path] = []
        self.index: list[Path] | None = None  # None = no search index (folder scan)

    def find_by_name(
        self, words: Sequence[str], roots: Sequence[Path], *, folders_only: bool
    ) -> list[Path] | None:
        return None if self.index is None else list(self.index)

    def open_path(self, path: Path, app: AppInfo | None = None) -> None:
        self.opened.append((path, app.name if app else None))

    def open_terminal(self, folder: Path) -> None:
        self.opened.append((folder, "Terminal"))

    def move_to_trash(self, path: Path) -> Path | None:
        import shutil

        self.trash.append(path)
        bin_dir = Path(os.environ["HOME"]) / ".Trash"
        bin_dir.mkdir(exist_ok=True)
        shutil.move(str(path), str(bin_dir / path.name))
        return bin_dir / path.name

    @property
    def apps(self) -> FakeAppControl:
        return self._apps

    def diagnostics(self) -> list[DiagnosticCheck]:
        return [DiagnosticCheck("Fake platform", True, "ok")]


class ScriptedConfirmer:
    """Answers confirmations from a fixed value and records the prompts."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def confirm(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return self.answer


def plan_json(action: str, message: str, *steps: tuple[str, dict[str, Any]]) -> str:
    """JSON for a plan, as the model would produce it."""
    return json.dumps(
        {
            "action": action,
            "message": message,
            "steps": [{"tool": tool, "arguments": args} for tool, args in steps],
        }
    )


class FakeLLM(LLMBackend):
    """Scripted model: returns queued replies and records every prompt."""

    name = "fake"

    def __init__(self, *replies: str, load_error: Exception | None = None) -> None:
        self.replies: deque[str] = deque(replies)
        self.load_error = load_error
        self.loaded = False
        self.loads = 0
        self.unloads = 0
        self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loads += 1
        self.loaded = True

    def unload(self) -> None:
        if self.loaded:
            self.unloads += 1
            self.loaded = False

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Completion:
        if not self.loaded:
            raise AssertionError("complete_json called before load()")
        self.calls.append((list(messages), schema))
        text = (
            self.replies.popleft() if self.replies else plan_json("respond", "No scripted reply.")
        )
        return Completion(text=text, prompt_tokens=120, completion_tokens=24, seconds=0.01)
