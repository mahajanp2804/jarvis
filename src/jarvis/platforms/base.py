"""Platform adapter interfaces.

Shared code (agent, dispatcher, tools) depends only on these abstractions.
Each operating system provides concrete adapters in its own module, and only
that module may import OS-specific libraries (PyObjC on macOS, pywin32 on
Windows).

Implemented so far: application control, diagnostics, process stats (Phase 1-2),
and the background-service hooks (Phase 3): event loop, global hotkey, dialogs
and service management, screen capture and text recognition (Phase 4), and
speech (Phase 5).
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
    from jarvis.voice.speech import SpeechOutput


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
class TextLine:
    """One line of recognised text; box is normalised (0-1), origin top-left."""

    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


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

    # -- files (Phase 4) ---------------------------------------------------------------

    def find_by_name(
        self, words: Sequence[str], roots: Sequence[Path], *, folders_only: bool
    ) -> list[Path] | None:
        """Indexed name search (all ``words`` in the name), or ``None`` if unavailable.

        Callers must still apply the path policy to every result.
        """
        return None

    def open_path(self, path: Path, app: AppInfo | None = None) -> None:
        """Open a file or folder with its default app, or with ``app``."""
        raise UnsupportedPlatformError(f"Opening files isn't implemented on {self.name}.")

    def open_terminal(self, folder: Path) -> None:
        """Open a terminal window in ``folder``."""
        raise UnsupportedPlatformError(f"Opening a terminal isn't implemented on {self.name}.")

    def move_to_trash(self, path: Path) -> Path | None:
        """Move to the Trash (recoverable). Returns the new location if known."""
        raise UnsupportedPlatformError(f"Moving to the Trash isn't implemented on {self.name}.")

    # -- screen (Phase 4, opt-in) ------------------------------------------------------

    def screen_capture_allowed(self) -> bool | None:
        """Whether the OS currently allows screen capture (``None`` = unknown)."""
        return None

    def capture_screen(self, dest: Path) -> None:
        """Save a screenshot of the main display to ``dest`` (PNG)."""
        raise UnsupportedPlatformError(f"Screen capture isn't implemented on {self.name}.")

    def recognize_text(self, image: Path) -> list[TextLine]:
        """On-device text recognition, lines ordered top to bottom."""
        raise UnsupportedPlatformError(f"Text recognition isn't implemented on {self.name}.")

    # -- speech (Phase 5, optional) ---------------------------------------------------

    def speech_output(self, voice: str | None = None, rate: int | None = None) -> SpeechOutput:
        """On-device text-to-speech (nothing runs until something is spoken)."""
        raise UnsupportedPlatformError(f"Spoken replies aren't implemented on {self.name}.")

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
