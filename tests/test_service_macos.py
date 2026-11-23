"""Launch Agent management with a fake ``launchctl`` (no real launchd changes)."""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from jarvis.platforms.macos_service import (
    LAUNCHCTL,
    SFLTOOL,
    LaunchAgentService,
    ServiceError,
    background_flags,
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
# Shape of real `sfltool dumpbtm` output (macOS 27), trimmed to three records.
BTM_DUMP = """ #1:
                 UUID: 11111111-2222-3333-4444-555555555555
                 Name: Google Chrome
          Disposition: [enabled, allowed, notified] (0xb)
           Identifier: 2.com.google.Chrome
 #2:
                 Name: Python
                 Type: legacy agent (0x10008)
          Disposition: [{flags}] (0x9)
           Identifier: 8.local.jarvis.assistant
                  URL: /Users/501/Library/LaunchAgents/local.jarvis.assistant.plist
      Executable Path: /Users/501/Desktop/Projects/jarvis/.venv/bin/python
 #3:
          Disposition: [disabled, disallowed, notified] (0x8)
           Identifier: 8.local.jarvis.assistant.other
"""

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
        self.background = "enabled, allowed, notified"  # what `sfltool dumpbtm` reports

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "shell" not in kwargs and kwargs.get("timeout")
        if argv[0] == SFLTOOL:
            assert argv[1:] == ["dumpbtm"]
            return subprocess.CompletedProcess(argv, 0, BTM_DUMP.format(flags=self.background), "")
        assert argv[0] == LAUNCHCTL
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


def python_in(folder: Path) -> Path:
    program = folder / ".venv" / "bin" / "python"
    program.parent.mkdir(parents=True)
    program.write_text("")
    return program


def test_start_kickstarts_or_bootstraps(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    with pytest.raises(ServiceError, match="isn't installed"):
        service.start()
    service.install([str(python_in(tmp_path))], {}, tmp_path / "s.log")
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


def test_background_flags_are_read_for_our_label_only() -> None:
    dump = BTM_DUMP.format(flags="enabled, disallowed, notified")
    flags = background_flags(dump, "local.jarvis.assistant")
    assert flags == {"enabled", "disallowed", "notified"}  # not the ".other" record
    assert background_flags(BTM_DUMP, "local.missing") is None
    assert background_flags("garbage", "local.jarvis.assistant") is None


def test_status_reports_the_program_it_runs(service: LaunchAgentService, tmp_path: Path) -> None:
    program = python_in(tmp_path / "Projects" / "jarvis")
    service.install([str(program), "-m", "jarvis", "daemon"], {}, tmp_path / "s.log")
    status = service.status()
    assert status.program == program
    assert service.problems(status) == []  # running and allowed


def test_switched_off_in_login_items_is_explained_and_respected(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    service.install([str(python_in(tmp_path))], {}, tmp_path / "s.log")
    launchctl.background = "enabled, disallowed, notified"  # the user switched it off
    launchctl.loaded = launchctl.running = False  # macOS unloaded it
    [problem] = service.problems()
    assert "Login Items & Extensions > Allow in the Background" in problem.message
    assert '"Python"' in problem.message and "Switch it back on" in problem.fix
    calls = len(launchctl.calls)
    with pytest.raises(ServiceError, match="switched off"):
        service.start()  # never overrides the user's switch
    assert len(launchctl.calls) == calls


def test_moved_folder_is_explained(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    program = python_in(tmp_path / "Project" / "jarvis")
    service.install([str(program)], {}, tmp_path / "s.log")
    program.parent.parent.parent.rename(tmp_path / "Project" / "renamed")
    launchctl.loaded = launchctl.running = False
    problems = service.problems()
    assert "no longer exists" in problems[0].message and "moved or renamed" in problems[0].message
    assert "setup_macos.sh" in problems[0].fix and "install_macos.sh" in problems[0].fix
    with pytest.raises(ServiceError, match="no longer exists"):
        service.start()


def test_stopped_service_just_needs_a_start(
    service: LaunchAgentService, launchctl: FakeLaunchctl, tmp_path: Path
) -> None:
    service.install([str(python_in(tmp_path))], {}, tmp_path / "s.log")
    service.stop()
    [problem] = service.problems()
    assert problem.message == "Jarvis is installed but not running." and not problem.blocks_start
    service.start()
    assert launchctl.calls[-1][0] == "kickstart"
