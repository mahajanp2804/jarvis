"""Tests against the real macOS system.

* ``integration`` tests are read-only (discover apps, check AppKit). They run by
  default on macOS and are skipped elsewhere.
* The ``gui`` test launches and quits Calculator. It only runs with
  ``JARVIS_RUN_GUI_TESTS=1`` and is skipped if Calculator is already open,
  so it never touches an app you are using.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from jarvis.api.schemas import ToolCall, ToolStatus
from jarvis.bootstrap import build_runtime
from jarvis.config import JarvisConfig

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@pytest.mark.integration
def test_real_discovery_finds_system_apps() -> None:
    from jarvis.platforms.macos import MacOSAppControl

    control = MacOSAppControl()
    apps = control.discover(control.default_search_paths())
    names = {a.name for a in apps}
    assert {"Calculator", "Safari"} & names, "expected standard macOS apps to be discovered"
    calculator = next(a for a in apps if a.name == "Calculator")
    assert calculator.bundle_id == "com.apple.calculator"


@pytest.mark.integration
def test_real_running_check_does_not_raise() -> None:
    from jarvis.platforms.macos import MacOSAppControl

    finder = MacOSAppControl().discover([Path("/System/Library/CoreServices")])
    finder_app = next(a for a in finder if a.bundle_id == "com.apple.finder")
    assert MacOSAppControl().is_running(finder_app) is True


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_GUI_TESTS") != "1", reason="set JARVIS_RUN_GUI_TESTS=1 to run"
)
def test_open_and_quit_calculator(tmp_path: Path) -> None:
    from jarvis.platforms.macos import MacOSPlatform
    from tests.fakes import ScriptedConfirmer

    platform = MacOSPlatform()
    config = JarvisConfig.model_validate({"general": {"log_dir": str(tmp_path)}})

    def runtime_with(answer: bool):  # type: ignore[no-untyped-def]
        return build_runtime(config, platform, ScriptedConfirmer(answer))

    runtime = runtime_with(True)
    calculator = runtime.catalog.resolve("calculator")
    if platform.apps.is_running(calculator):
        pytest.skip("Calculator is already running; not touching it")

    opened = runtime.dispatcher.execute(ToolCall(tool="apps.open", arguments={"name": "calc"}))
    assert opened.status is ToolStatus.SUCCESS, opened.message
    assert opened.verified is True

    declined = runtime_with(False)
    refused = declined.dispatcher.execute(
        ToolCall(tool="apps.quit", arguments={"name": "Calculator"})
    )
    assert refused.status is ToolStatus.CANCELLED
    assert platform.apps.is_running(calculator) is True
    declined.close()

    closed = runtime.dispatcher.execute(
        ToolCall(tool="apps.quit", arguments={"name": "Calculator"})
    )
    assert closed.status is ToolStatus.SUCCESS, closed.message
    assert closed.verified is True
    runtime.close()


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_GUI_TESTS") != "1", reason="set JARVIS_RUN_GUI_TESTS=1 to run"
)
def test_gui_service_closes_open_dialog_on_sigterm() -> None:
    """Start the real GUI service, open its dialog, then SIGTERM while it is open."""
    import signal
    import subprocess
    import sys
    import tempfile
    import time

    from jarvis.runtime.control import send_request

    with tempfile.TemporaryDirectory(prefix="jv", dir="/tmp") as home:
        env = {**os.environ, "HOME": home, "JARVIS_AI_ENABLED": "0", "JARVIS_DRY_RUN": "1"}
        env.pop("JARVIS_LOG_DIR", None)
        env.pop("JARVIS_CONFIG", None)
        # Another shortcut than the default, so a running Jarvis service doesn't clash.
        config = Path(home) / "config.toml"
        config.write_text('[service]\nhotkey = "ctrl+option+cmd+f12"\n')
        proc = subprocess.Popen(
            [sys.executable, "-m", "jarvis", "--config", str(config), "daemon"], env=env
        )
        sock = Path(home) / "Library/Application Support/Jarvis/run/jarvis.sock"
        log = Path(home) / "Library/Logs/Jarvis/jarvis.log"
        try:
            deadline = time.monotonic() + 15
            while not sock.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            status = send_request(sock, {"cmd": "status"})
            assert status["mode"] == "gui" and status["hotkey"] == "⌃⌥⌘F12"
            assert send_request(sock, {"cmd": "activate"})["ok"]
            time.sleep(1.0)  # dialog is now open on screen
            started = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=10) == 0
            assert time.monotonic() - started < 3
            assert "Closing an open dialog for shutdown" in log.read_text()
        finally:
            if proc.poll() is None:
                proc.kill()
