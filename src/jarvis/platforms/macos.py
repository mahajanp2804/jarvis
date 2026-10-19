"""macOS adapters (tested on macOS 27 / Apple Silicon).

Application control:

* Discovery scans standard install folders for ``.app`` bundles and reads
  ``Contents/Info.plist`` with the standard-library :mod:`plistlib`.
* Launching uses ``/usr/bin/open -a <bundle path>`` with a fixed argument list
  (no shell), so only bundles that discovery found can be started.
* Running-state checks and graceful quitting use AppKit's
  ``NSRunningApplication`` through PyObjC. ``terminate()`` sends a normal quit
  request, so apps can still prompt to save, and it does not need the
  Automation (Apple Events) privacy permission that ``osascript`` would.

PyObjC is imported lazily so commands that don't need it stay fast.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import plistlib
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.errors import OSPermissionError, ToolExecutionError, UnsupportedPlatformError
from jarvis.platforms.base import (
    AppControlAdapter,
    AppInfo,
    DiagnosticCheck,
    PlatformAdapter,
    ProcessStats,
    ServiceManager,
    TextLine,
)

if TYPE_CHECKING:
    from jarvis.runtime.event_loop import EventLoop
    from jarvis.runtime.hotkeys import HotkeyListener
    from jarvis.ui.base import UserInterface

log = logging.getLogger(__name__)

OPEN_BIN = "/usr/bin/open"
SCREENCAPTURE_BIN = "/usr/sbin/screencapture"
SCREEN_PERMISSION_HINT = (
    "Allow Screen Recording for the app running Jarvis (Terminal for `jarvis run`, Python "
    "for the background service) in System Settings > Privacy & Security > "
    "Screen & System Audio Recording, then restart Jarvis."
)
MDFIND_BIN = "/usr/bin/mdfind"
SYSCTL_BIN = "/usr/sbin/sysctl"
_SUBPROCESS_TIMEOUT = 15.0

DEFAULT_ALIASES: dict[str, str] = {
    "chrome": "Google Chrome",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "code": "Visual Studio Code",
    "settings": "System Settings",
    "system preferences": "System Settings",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
    "outlook": "Microsoft Outlook",
    "teams": "Microsoft Teams",
    "calc": "Calculator",
}

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _load_appkit() -> Any:
    try:
        import AppKit
    except ImportError as exc:  # pragma: no cover - exercised only without PyObjC
        raise UnsupportedPlatformError(
            "PyObjC (pyobjc-framework-Cocoa) is not installed, so Jarvis cannot "
            "check or quit running applications.",
            hint="Inside the Jarvis virtualenv run: pip install -e .",
        ) from exc
    return AppKit


def read_app_bundle(bundle: Path) -> AppInfo:
    """Build an :class:`AppInfo` from a ``.app`` bundle, tolerating a broken Info.plist."""
    info: dict[str, Any] = {}
    try:
        with (bundle / "Contents" / "Info.plist").open("rb") as fh:
            loaded = plistlib.load(fh)
        if isinstance(loaded, dict):
            info = loaded
    except Exception as exc:
        # Third-party bundles can ship malformed plists (plistlib may raise
        # OSError, InvalidFileException, ValueError or expat.ExpatError). One bad
        # bundle must never break discovery; fall back to the folder name.
        log.debug("Unreadable Info.plist in %s: %s: %s", bundle, type(exc).__name__, exc)

    name = bundle.stem
    alt_names: list[str] = []
    for key in ("CFBundleDisplayName", "CFBundleName"):
        value = info.get(key)
        if isinstance(value, str) and value.strip() not in ("", name, *alt_names):
            alt_names.append(value.strip())
    bundle_id = info.get("CFBundleIdentifier")
    return AppInfo(
        name=name,
        path=bundle,
        bundle_id=bundle_id if isinstance(bundle_id, str) and bundle_id else None,
        alt_names=tuple(alt_names),
    )


class MacOSAppControl(AppControlAdapter):
    """Application control backed by LaunchServices (``open``) and AppKit."""

    def __init__(
        self,
        runner: Runner = subprocess.run,
        appkit_loader: Callable[[], Any] = _load_appkit,
    ) -> None:
        self._run = runner
        self._load_appkit = appkit_loader

    def default_search_paths(self) -> list[Path]:
        return [
            Path("/Applications"),
            Path("/Applications/Utilities"),
            Path("/System/Applications"),
            Path("/System/Applications/Utilities"),
            Path.home() / "Applications",
        ]

    def default_aliases(self) -> dict[str, str]:
        return dict(DEFAULT_ALIASES)

    def discover(self, search_paths: Sequence[Path]) -> list[AppInfo]:
        """Find ``.app`` bundles directly inside each path, or one folder deeper.

        The one-level descent covers vendor folders such as
        ``/Applications/Adobe Photoshop 2026/``. Bundles are never descended into.
        Duplicates (same bundle ID or same path) keep the first occurrence, so the
        order of ``search_paths`` sets precedence.
        """
        apps: list[AppInfo] = []
        seen_paths: set[Path] = set()
        seen_ids: set[str] = set()

        def add(bundle: Path) -> None:
            real = bundle.resolve()
            if real in seen_paths:
                return
            app = read_app_bundle(bundle)
            if app.bundle_id and app.bundle_id in seen_ids:
                return
            seen_paths.add(real)
            if app.bundle_id:
                seen_ids.add(app.bundle_id)
            apps.append(app)

        for root in search_paths:
            for entry in _safe_listdir(root):
                if entry.suffix == ".app":
                    add(entry)
                elif entry.is_dir() and not entry.name.startswith("."):
                    for child in _safe_listdir(entry):
                        if child.suffix == ".app":
                            add(child)
        return apps

    def launch(self, app: AppInfo) -> None:
        argv = [OPEN_BIN, "-a", str(app.path)]
        try:
            proc = self._run(argv, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(f"Timed out asking macOS to open {app.name}.") from exc
        except OSError as exc:
            raise ToolExecutionError(f"Could not run {OPEN_BIN}: {exc.strerror}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
            raise ToolExecutionError(
                f"macOS could not open {app.name}: {detail}",
                hint="If this is the app's first launch, open it once from Finder so "
                "macOS can verify it (Gatekeeper).",
            )

    def ensure_appkit(self) -> None:
        """Raise :class:`UnsupportedPlatformError` if PyObjC's AppKit is unavailable."""
        self._load_appkit()

    def _running_instances(self, app: AppInfo) -> list[Any]:
        if not app.bundle_id:
            return []
        appkit = self._load_appkit()
        instances = appkit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(
            app.bundle_id
        )
        return [inst for inst in instances if not inst.isTerminated()]

    def is_running(self, app: AppInfo) -> bool | None:
        if not app.bundle_id:
            return None
        return bool(self._running_instances(app))

    def quit(self, app: AppInfo) -> None:
        if not app.bundle_id:
            raise ToolExecutionError(
                f"Cannot quit {app.name}: its bundle has no identifier.",
                hint="Quit it manually.",
            )
        instances = self._running_instances(app)
        if not instances:
            return
        accepted = [bool(inst.terminate()) for inst in instances]
        if not any(accepted):
            raise ToolExecutionError(
                f"macOS did not accept the quit request for {app.name}.",
                hint="The app may be busy or showing a modal dialog; quit it manually.",
            )


def _safe_listdir(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except FileNotFoundError:
        return []
    except (PermissionError, NotADirectoryError) as exc:
        log.debug("Skipping %s: %s", path, exc)
        return []


class _RUsageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` from <sys/resource.h>."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_RUSAGE_INFO_V2 = 2


class _MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _mach_ticks_to_seconds() -> float:
    """rusage CPU times are mach ticks (125/3 ns each on Apple Silicon)."""
    timebase = _MachTimebase()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_timebase_info(ctypes.byref(timebase))
    return float(timebase.numer) / float(timebase.denom) / 1e9


def read_process_stats(pid: int | None = None) -> ProcessStats | None:
    """Memory, CPU time and wakeups for ``pid`` (default: this process).

    Uses ``proc_pid_rusage``, which works for any process of the same user.
    """
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    except OSError:
        return None
    info = _RUsageInfoV2()
    if libproc.proc_pid_rusage(pid or os.getpid(), _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        return None
    return ProcessStats(
        rss_bytes=int(info.ri_resident_size),
        footprint_bytes=int(info.ri_phys_footprint),
        wakeups=int(info.ri_pkg_idle_wkups + info.ri_interrupt_wkups),
        cpu_seconds=float(info.ri_user_time + info.ri_system_time) * _mach_ticks_to_seconds(),
    )


class MacOSPlatform(PlatformAdapter):
    """macOS implementation of :class:`PlatformAdapter`."""

    name = "macOS"

    def __init__(self, apps: MacOSAppControl | None = None, runner: Runner = subprocess.run):
        self._apps = apps or MacOSAppControl(runner=runner)
        self._run = runner

    @property
    def apps(self) -> MacOSAppControl:
        return self._apps

    def diagnostics(self) -> list[DiagnosticCheck]:
        checks = [
            DiagnosticCheck("macOS version", True, platform.mac_ver()[0] or "unknown"),
        ]

        machine = platform.machine()
        translated = self._rosetta_translated()
        if translated:
            checks.append(
                DiagnosticCheck(
                    "Native arm64 Python",
                    False,
                    "running under Rosetta (x86_64); use an arm64 Python such as "
                    "/opt/homebrew/bin/python3 for Metal-accelerated inference",
                )
            )
        else:
            checks.append(DiagnosticCheck("Native arm64 Python", machine == "arm64", machine))

        try:
            self._apps.ensure_appkit()
            checks.append(DiagnosticCheck("PyObjC AppKit", True, "available"))
        except UnsupportedPlatformError as exc:
            checks.append(DiagnosticCheck("PyObjC AppKit", False, exc.message))

        checks.append(DiagnosticCheck("LaunchServices 'open'", Path(OPEN_BIN).exists(), OPEN_BIN))
        return checks

    def process_stats(self, pid: int | None = None) -> ProcessStats | None:
        return read_process_stats(pid)

    def find_by_name(
        self, words: Sequence[str], roots: Sequence[Path], *, folders_only: bool
    ) -> list[Path] | None:
        """Spotlight name search: fast (indexed) and no polling.

        ``words`` are reduced to letters and digits, so model-supplied text
        can't alter the Spotlight query.
        """
        safe = [re.sub(r"[^0-9a-z]", "", w.lower()) for w in words]
        safe = [w for w in safe if w]
        if not safe:
            return []
        clauses = [f'kMDItemFSName == "*{w}*"cd' for w in safe]
        if folders_only:
            clauses.append('kMDItemContentType == "public.folder"')
        query = " && ".join(clauses)
        found: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            try:
                proc = self._run(
                    [MDFIND_BIN, "-onlyin", str(root), query],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            if proc.returncode != 0:
                return None
            found.extend(Path(line) for line in proc.stdout.splitlines() if line.strip())
        return list(dict.fromkeys(found))

    def open_path(self, path: Path, app: AppInfo | None = None) -> None:
        argv = [OPEN_BIN, "-a", str(app.path), str(path)] if app else [OPEN_BIN, str(path)]
        try:
            proc = self._run(argv, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolExecutionError(f"Could not open {path.name}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
            raise ToolExecutionError(f"macOS could not open {path.name}: {detail}")

    def screen_capture_allowed(self) -> bool | None:
        try:
            import Quartz
        except ImportError:
            return None
        # Preflight only checks; it never shows the permission prompt.
        return bool(Quartz.CGPreflightScreenCaptureAccess())

    def capture_screen(self, dest: Path) -> None:
        if self.screen_capture_allowed() is False:
            raise OSPermissionError(
                "macOS doesn't allow Jarvis to capture the screen.", hint=SCREEN_PERMISSION_HINT
            )
        argv = [SCREENCAPTURE_BIN, "-x", "-m", "-t", "png", str(dest)]  # -x: no sound
        try:
            proc = self._run(argv, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolExecutionError(f"Screen capture failed: {exc}") from exc
        detail = (proc.stderr or proc.stdout).strip()
        if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
            if "could not create image" in detail or not detail:
                raise OSPermissionError(
                    "macOS blocked the screen capture.", hint=SCREEN_PERMISSION_HINT
                )
            raise ToolExecutionError(f"Screen capture failed: {detail}")

    def recognize_text(self, image: Path) -> list[TextLine]:
        try:
            import Vision
            from Foundation import NSURL
        except ImportError as exc:
            raise UnsupportedPlatformError(
                "Text recognition needs Apple's Vision framework bindings.",
                hint='In the Jarvis virtualenv run: pip install -e ".[screen]"',
            ) from exc
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
            NSURL.fileURLWithPath_(str(image)), None
        )
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            reason = error.localizedDescription() if error is not None else "unknown error"
            raise ToolExecutionError(f"Text recognition failed: {reason}")
        lines: list[TextLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            box = observation.boundingBox()  # normalised, origin bottom-left
            lines.append(
                TextLine(
                    text=str(candidates[0].string()),
                    confidence=float(candidates[0].confidence()),
                    x=float(box.origin.x),
                    y=float(1.0 - box.origin.y - box.size.height),
                    width=float(box.size.width),
                    height=float(box.size.height),
                )
            )
        lines.sort(key=lambda line: (round(line.y, 2), line.x))
        return lines

    def open_terminal(self, folder: Path) -> None:
        self.open_path(
            folder, AppInfo("Terminal", Path("/System/Applications/Utilities/Terminal.app"))
        )

    def move_to_trash(self, path: Path) -> Path | None:
        from Foundation import NSURL, NSFileManager

        ok, result_url, error = (
            NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
                NSURL.fileURLWithPath_(str(path)), None, None
            )
        )
        if not ok:
            reason = error.localizedDescription() if error is not None else "unknown error"
            raise ToolExecutionError(f"Could not move {path.name} to the Trash: {reason}")
        return Path(str(result_url.path())) if result_url is not None else None

    def event_loop(self, *, headless: bool) -> EventLoop:
        if headless:
            return super().event_loop(headless=True)
        from jarvis.platforms.macos_gui import AppKitEventLoop

        return AppKitEventLoop()

    def hotkey_listener(self, loop: EventLoop) -> HotkeyListener:
        from jarvis.platforms.macos_gui import CarbonHotkeyListener

        return CarbonHotkeyListener(loop)

    def user_interface(self) -> UserInterface:
        from jarvis.platforms.macos_gui import AlertUI

        return AlertUI()

    def service_manager(self, label: str) -> ServiceManager:
        from jarvis.platforms.macos_service import LaunchAgentService

        return LaunchAgentService(label, runner=self._run)

    def _rosetta_translated(self) -> bool:
        try:
            proc = self._run(
                [SYSCTL_BIN, "-in", "sysctl.proc_translated"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.stdout.strip() == "1"
