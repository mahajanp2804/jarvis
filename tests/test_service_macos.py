"""Launch Agent management with a fake ``launchctl`` (no real launchd changes)."""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from jarvis.platforms.macos_service import (
    LAUNCHCTL,
    LaunchAgentService,
    ServiceError,
    build_plist,
)

PRINT_RUNNING = """gui/501/local.jarvis.assistant = {
\tactive count = 1
\tpath = /Users/me/Library/LaunchAgents/local.jarvis.assistant.plist
\tstate = running
\tprogram = /x/.venv/bin/python
\tpid = 4242
\tlast exit code = (never exited)
}"""
PRINT_STOPPED = (
    PRINT_RUNNING.replace("state = running", "state = not running")
    .replace("\tpid = 4242\n", "")
    .replace("(never exited)", "0")
)


class FakeLaunchctl:
    """Simulates launchd state transitions for the commands Jarvis uses."""

    def __init__(self) -> None:
        self.loaded = False
        self.running = False
        self.calls: list[list[str]] = []
        self.fail: str | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[0] == LAUNCHCTL
        assert "shell" not in kwargs and kwargs.get("timeout")
        command = argv[1]
        if command != "print":
            self.calls.append(argv[1:])
        if command == self.fail:
            return subprocess.CompletedProcess(argv, 5, "", "Boom: input/output error")
        if command == "print":
            if not self.loaded:
                return subprocess.CompletedProcess(argv, 113, "", "Could not find service")
            return subprocess.CompletedProcess(
                argv, 0, PRINT_RUNNING if self.running else PRINT_STOPPED, ""
            )
        if command == "bootstrap":
            self.loaded = self.running = True
        elif command == "bootout":
            self.loaded = self.running = False
        elif command == "kickstart":
            self.running = True
        elif command == "kill":
            self.running = False
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def launchctl() -> FakeLaunchctl:
    return FakeLaunchctl()


@pytest.fixture
def service(tmp_path: Path, launchctl: FakeLaunchctl) -> LaunchAgentService:
    return LaunchAgentService(
        "local.jarvis.assistant",
        runner=launchctl,
        agents_dir=tmp_path / "LaunchAgents",
        uid=501,
        sleep=lambda _s: None,
    )


def test_plist_contents(tmp_path: Path) -> None:
    plist = build_plist(
        "local.jarvis.assistant",
        ["/venv/bin/python", "-m", "jarvis", "daemon"],
        {"PATH": "/usr/bin:/bin"},
        tmp_path / "service.log",
        tmp_path,
    )
    assert plist["ProgramArguments"] == ["/venv/bin/python", "-m", "jarvis", "daemon"]
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] == {"Crashed": True}  # clean stops stay stopped
    assert plist["LimitLoadToSessionType"] == "Aqua"
    assert plist["ExitTimeOut"] >= 15
    # launchd also throttles manual starts by this interval; keep restarts quick.
    assert plist["ThrottleInterval"] == 10
    plistlib.dumps(plist)  # serialisable


def test_install_writes_plist_and_bootstraps(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    path = service.install(
        ["/py", "-m", "jarvis", "daemon"], {"PATH": "/usr/bin"}, tmp_path / "l" / "s.log"
    )
    assert path == tmp_path / "LaunchAgents" / "local.jarvis.assistant.plist"
    data = plistlib.loads(path.read_bytes())
    assert data["Label"] == "local.jarvis.assistant"
    assert launchctl.calls == [["bootstrap", "gui/501", str(path)]]
    status = service.status()
    assert status.installed and status.loaded and status.running and status.pid == 4242


def test_reinstall_boots_out_first(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    service.install(["/py"], {}, tmp_path / "s.log")
    service.install(["/py"], {}, tmp_path / "s.log")
    assert [c[0] for c in launchctl.calls] == ["bootstrap", "bootout", "bootstrap"]


def test_stop_is_a_graceful_sigterm(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    service.install(["/py"], {}, tmp_path / "s.log")
    service.stop()
    assert launchctl.calls[-1] == ["kill", "SIGTERM", "gui/501/local.jarvis.assistant"]
    status = service.status()
    assert status.loaded and not status.running and status.last_exit == "0"
    service.stop()  # already stopped: no-op
    assert launchctl.calls[-1][0] == "kill" and len(launchctl.calls) == 2


def test_start_kickstarts_or_bootstraps(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    with pytest.raises(ServiceError, match="isn't installed"):
        service.start()
    service.install(["/py"], {}, tmp_path / "s.log")
    service.stop()
    service.start()
    assert launchctl.calls[-1] == ["kickstart", "gui/501/local.jarvis.assistant"]
    launchctl.loaded = launchctl.running = False  # e.g. after `launchctl bootout`
    service.start()
    assert launchctl.calls[-1][0] == "bootstrap"


def test_uninstall(service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path) -> None:
    path = service.install(["/py"], {}, tmp_path / "s.log")
    service.uninstall()
    assert not path.exists()
    assert launchctl.calls[-1][0] == "bootout"
    assert not service.status().installed
    service.uninstall()  # idempotent


def test_launchctl_failure_is_reported(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    launchctl.fail = "bootstrap"
    with pytest.raises(ServiceError, match="bootstrap failed: Boom"):
        service.install(["/py"], {}, tmp_path / "s.log")
