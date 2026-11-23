"""Start-at-login via a per-user launchd Launch Agent.

The agent file lives at ``~/Library/LaunchAgents/<label>.plist``:

* ``RunAtLoad``: start at login (and when installed).
* ``KeepAlive = {Crashed: true}``: restart only after a *crash*. A clean exit
  (``jarvis service stop``, or SIGTERM from logout) stays stopped until the
  next login or ``jarvis service start``.
* ``LimitLoadToSessionType = Aqua``: only in a graphical login session, where
  the hotkey and dialogs work.
* ``ProcessType = Interactive``: the agent responds to the user, so it must
  not be throttled like a background job. Idle, it uses no CPU anyway.
* ``ExitTimeOut``: launchd waits this long after SIGTERM before SIGKILL. Jarvis
  shuts down in well under that (measured in the README).
* ``ThrottleInterval``: launchd starts the job at most once every 10 s. This
  guards against crash loops and also delays a quick stop/start, which
  ``jarvis service restart`` waits out.

Since macOS 13, the user can switch the agent off in System Settings > General >
Login Items & Extensions ("Allow in the Background"); it's listed as "Python"
because Jarvis is a Python program. macOS then unloads it. Jarvis reads that
switch from the background-task database (``sfltool dumpbtm``, no admin rights
needed) to explain why it isn't running, and never overrides it.

All ``launchctl`` and ``sfltool`` calls use fixed argument lists, with no shell.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from jarvis.errors import JarvisError
from jarvis.platforms.base import ServiceManager, ServiceStatus

LAUNCHCTL = "/bin/launchctl"
SFLTOOL = "/usr/bin/sfltool"
_BTM_RECORD = re.compile(r"^\s*#\d+:\s*$", re.MULTILINE)
EXIT_TIMEOUT_SECONDS = 20
# launchd starts a job at most once per this many seconds (crash-loop guard).
# It also delays manual starts, so `restart` must wait out the window.
THROTTLE_SECONDS = 10
_NOT_LOADED = 113  # "Could not find service" from `launchctl print`

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class ServiceError(JarvisError):
    """launchd refused an operation."""


def background_flags(dump: str, label: str) -> set[str] | None:
    """Disposition flags (enabled, allowed, ...) of ``label`` in ``sfltool dumpbtm`` output."""
    identifier = re.compile(rf"^\s*Identifier:\s*\d+\.{re.escape(label)}\s*$", re.MULTILINE)
    for record in _BTM_RECORD.split(dump):
        if identifier.search(record):
            match = re.search(r"^\s*Disposition:\s*\[([^\]]*)\]", record, re.MULTILINE)
            return {flag.strip() for flag in match.group(1).split(",")} if match else None
    return None


def build_plist(
    label: str,
    program: Sequence[str],
    environment: dict[str, str],
    log_file: Path,
    working_directory: Path,
) -> dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": list(program),
        "RunAtLoad": True,
        "KeepAlive": {"Crashed": True},
        "LimitLoadToSessionType": "Aqua",
        "ProcessType": "Interactive",
        "ThrottleInterval": THROTTLE_SECONDS,
        "ExitTimeOut": EXIT_TIMEOUT_SECONDS,
        "WorkingDirectory": str(working_directory),
        "StandardOutPath": str(log_file),
        "StandardErrorPath": str(log_file),
        "EnvironmentVariables": dict(environment),
    }


class LaunchAgentService(ServiceManager):
    """Manage the Jarvis Launch Agent with ``launchctl``."""

    background_setting = (
        "System Settings > General > Login Items & Extensions > Allow in the Background "
        '(listed as "Python")'
    )

    def __init__(
        self,
        label: str,
        *,
        runner: Runner = subprocess.run,
        agents_dir: Path | None = None,
        uid: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.label = label
        self.plist_path = (
            agents_dir or Path.home() / "Library" / "LaunchAgents"
        ) / f"{label}.plist"
        self._uid = os.getuid() if uid is None else uid
        self._run = runner
        self._sleep = sleep

    @property
    def domain(self) -> str:
        return f"gui/{self._uid}"

    @property
    def target(self) -> str:
        return f"{self.domain}/{self.label}"

    def install(self, program: Sequence[str], environment: dict[str, str], log_file: Path) -> Path:
        log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        working_directory = log_file.parent
        plist = build_plist(self.label, program, environment, log_file, working_directory)
        if self.status().loaded:
            self._launchctl("bootout", self.target)
            self._wait_until(lambda s: not s.loaded)
        self.plist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.plist_path.open("wb") as fh:
            plistlib.dump(plist, fh)
        self.plist_path.chmod(0o644)
        self._launchctl("bootstrap", self.domain, str(self.plist_path))
        return self.plist_path

    def uninstall(self) -> None:
        if self.status().loaded:
            self._launchctl("bootout", self.target)
            self._wait_until(lambda s: not s.loaded)
        self.plist_path.unlink(missing_ok=True)

    def start(self) -> None:
        status = self.status()
        if not status.installed:
            raise ServiceError(
                "The Jarvis service isn't installed.", hint="Run `jarvis service install`."
            )
        if not status.loaded:
            # Explain instead of failing obscurely, and never override the user's
            # "Allow in the Background" switch.
            blocking = [p for p in self.problems(status) if p.blocks_start]
            if blocking:
                raise ServiceError(blocking[0].message, hint=blocking[0].fix)
            self._launchctl("bootstrap", self.domain, str(self.plist_path))
        elif not status.running:
            self._launchctl("kickstart", self.target)

    def stop(self) -> None:
        status = self.status()
        if not status.running:
            return
        # SIGTERM -> graceful shutdown -> clean exit, which KeepAlive {Crashed}
        # does not restart. The agent stays loaded for the next login.
        self._launchctl("kill", "SIGTERM", self.target)
        if not self._wait_until(lambda s: not s.running, timeout=EXIT_TIMEOUT_SECONDS + 5):
            raise ServiceError("Jarvis did not stop in time; launchd will force it to quit.")

    def status(self) -> ServiceStatus:
        installed = self.plist_path.is_file()
        program = self._program() if installed else None
        proc = self._run(
            [LAUNCHCTL, "print", self.target], capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            return ServiceStatus(
                installed, False, False, definition=self.plist_path, program=program
            )
        text = proc.stdout
        state = _field(text, "state")
        pid = _field(text, "pid")
        return ServiceStatus(
            installed=installed,
            loaded=True,
            running=state == "running",
            pid=int(pid) if pid and pid.isdigit() else None,
            last_exit=_field(text, "last exit code"),
            definition=self.plist_path,
            program=program,
        )

    def allowed_to_run(self) -> bool | None:
        """The "Allow in the Background" switch (~2.5 s to read; only to explain problems)."""
        try:
            proc = self._run([SFLTOOL, "dumpbtm"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        flags = background_flags(proc.stdout, self.label) if proc.returncode == 0 else None
        if flags is None:
            return None
        if "disallowed" in flags:
            return False
        return True if "allowed" in flags else None

    def _program(self) -> Path | None:
        try:
            with self.plist_path.open("rb") as fh:
                arguments = plistlib.load(fh).get("ProgramArguments") or []
        except (OSError, plistlib.InvalidFileException, ValueError):
            return None
        return Path(arguments[0]) if arguments and isinstance(arguments[0], str) else None

    # -- internals -------------------------------------------------------------------

    def _launchctl(self, *args: str) -> None:
        proc = self._run([LAUNCHCTL, *args], capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit code {proc.returncode}"
            raise ServiceError(f"launchctl {args[0]} failed: {detail}")

    def _wait_until(self, predicate: Callable[[ServiceStatus], bool], timeout: float = 10) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if predicate(self.status()):
                return True
            if time.monotonic() >= deadline:
                return False
            self._sleep(0.25)


def _field(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(name)} = (.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None
