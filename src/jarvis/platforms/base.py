"""Platform adapter interfaces.

Shared code (agent, dispatcher, tools) depends only on these abstractions.
Each operating system provides concrete adapters in its own module, and only
that module may import OS-specific libraries (PyObjC on macOS, pywin32 on
Windows).

Implemented so far: application control, diagnostics, process stats (Phase 1-2),
and the background-service hooks (Phase 3): event loop, global hotkey, dialogs
and service management. Speech, file and browser adapters are added in the
phases that implement them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis.errors import UnsupportedPlatformError

if TYPE_CHECKING:
    from jarvis.runtime.event_loop import EventLoop
    from jarvis.runtime.hotkeys import HotkeyListener
    from jarvis.ui.base import UserInterface


@dataclass(frozen=True)
class AppInfo:
    """An installed application Jarvis is able to launch."""

    name: str  # name the user sees, e.g. "Visual Studio Code"
    path: Path  # bundle (macOS) or executable/shortcut (Windows)
    bundle_id: str | None = None  # macOS bundle ID; Windows AppUserModelID later
    alt_names: tuple[str, ...] = field(default_factory=tuple)  # e.g. CFBundleName "Code"

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.alt_names)


@dataclass(frozen=True)
class DiagnosticCheck:
    """One line of ``jarvis doctor`` output."""

    name: str
    ok: bool | None  # None = informational only
    detail: str


@dataclass(frozen=True)
class ProcessStats:
    """Resource usage of the current process (for benchmarks and ``jarvis doctor``)."""

    rss_bytes: int  # resident memory, including memory-mapped model pages
    footprint_bytes: int | None  # macOS "Memory" in Activity Monitor (dirty + compressed)
    wakeups: int | None  # CPU wakeups since process start, an energy-use proxy
    cpu_seconds: float | None = None  # user + system CPU time since process start


class AppControlAdapter(ABC):
    """Discover, launch and quit applications on one operating system."""

    @abstractmethod
    def default_search_paths(self) -> list[Path]:
        """Standard install locations to scan when the user configured none."""

    @abstractmethod
    def default_aliases(self) -> dict[str, str]:
        """Built-in alternative names, lower-case alias -> application name."""

    @abstractmethod
    def discover(self, search_paths: Sequence[Path]) -> list[AppInfo]:
        """Return installed applications found under ``search_paths``."""

    @abstractmethod
    def launch(self, app: AppInfo) -> None:
        """Start (or bring to front) ``app``. Raise a JarvisError on failure."""

    @abstractmethod
    def is_running(self, app: AppInfo) -> bool | None:
        """Whether ``app`` is running; ``None`` if this cannot be determined."""

    @abstractmethod
    def quit(self, app: AppInfo) -> None:
        """Ask ``app`` to quit gracefully (it may prompt to save). Never force-kill."""


@dataclass(frozen=True)
class ServiceStatus:
    """State of the background service as the OS service manager sees it."""

    installed: bool
    loaded: bool
    running: bool
    pid: int | None = None
    last_exit: str | None = None
    definition: Path | None = None  # e.g. the Launch Agent plist


class ServiceManager(ABC):
    """Start-at-login service lifecycle (launchd on macOS; Task Scheduler on Windows later)."""

    @abstractmethod
    def install(self, program: Sequence[str], environment: dict[str, str], log_file: Path) -> Path:
        """Register the service to start at login and start it now. Returns the definition."""

    @abstractmethod
    def uninstall(self) -> None:
        """Stop the service and remove its registration (keeps config, logs, models)."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """Ask the service to shut down gracefully."""

    @abstractmethod
    def status(self) -> ServiceStatus: ...


class PlatformAdapter(ABC):
    """Bundle of OS-specific adapters."""

    name: str

    @property
    @abstractmethod
    def apps(self) -> AppControlAdapter:
        """Application control for this OS."""

    @abstractmethod
    def diagnostics(self) -> list[DiagnosticCheck]:
        """OS-specific environment checks for ``jarvis doctor``."""

    def process_stats(self, pid: int | None = None) -> ProcessStats | None:
        """Memory/CPU/wakeups of ``pid`` (default: this process); ``None`` if unsupported."""
        return None

    # -- background service (Phase 3) -------------------------------------------------

    def event_loop(self, *, headless: bool) -> EventLoop:
        """Main-thread loop for the service. Headless needs no window server."""
        from jarvis.runtime.event_loop import QueueEventLoop

        if not headless:
            raise UnsupportedPlatformError(
                f"The graphical service isn't implemented on {self.name}.",
                hint="Run `jarvis daemon --headless`.",
            )
        return QueueEventLoop()

    def hotkey_listener(self, loop: EventLoop) -> HotkeyListener:
        raise UnsupportedPlatformError(f"Global shortcuts aren't implemented on {self.name}.")

    def user_interface(self) -> UserInterface:
        raise UnsupportedPlatformError(f"Dialogs aren't implemented on {self.name}.")

    def service_manager(self, label: str) -> ServiceManager:
        raise UnsupportedPlatformError(
            f"Start-at-login isn't implemented on {self.name}.",
            hint="See docs/WINDOWS.md for the plan.",
        )
