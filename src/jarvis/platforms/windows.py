"""Windows adapters: NOT IMPLEMENTED.

This module exists so the platform boundary is visible and so running Jarvis on
Windows fails with a clear message instead of an ImportError deep in a tool.
Nothing here has been run on Windows. See ``docs/WINDOWS.md`` for the
implementation plan (Start Menu discovery, ``ShellExecuteW`` launch,
``WM_CLOSE`` graceful quit, Task Scheduler start-up).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from jarvis.errors import UnsupportedPlatformError
from jarvis.platforms.base import AppControlAdapter, AppInfo, DiagnosticCheck, PlatformAdapter

_HINT = "Windows support is planned; see docs/WINDOWS.md."


def _unsupported(feature: str) -> NoReturn:
    raise UnsupportedPlatformError(f"{feature} is not implemented on Windows yet.", hint=_HINT)


class WindowsAppControl(AppControlAdapter):
    """Placeholder: every operation raises :class:`UnsupportedPlatformError`."""

    def default_search_paths(self) -> list[Path]:
        return []

    def default_aliases(self) -> dict[str, str]:
        return {}

    def discover(self, search_paths: Sequence[Path]) -> list[AppInfo]:
        _unsupported("Application discovery")

    def launch(self, app: AppInfo) -> None:
        _unsupported("Launching applications")

    def is_running(self, app: AppInfo) -> bool | None:
        _unsupported("Checking running applications")

    def quit(self, app: AppInfo) -> None:
        _unsupported("Quitting applications")


class WindowsPlatform(PlatformAdapter):
    """Placeholder Windows platform; diagnostics report that it is unsupported."""

    name = "Windows"

    def __init__(self) -> None:
        self._apps = WindowsAppControl()

    @property
    def apps(self) -> WindowsAppControl:
        return self._apps

    def diagnostics(self) -> list[DiagnosticCheck]:
        return [DiagnosticCheck("Windows support", False, "not implemented yet. " + _HINT)]
