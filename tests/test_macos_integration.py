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


@pytest.mark.integration
def test_real_ocr_on_a_generated_image(tmp_path: Path) -> None:
    """Apple Vision reads text from an image drawn here (no screen involved)."""
    pytest.importorskip("Vision")
    from AppKit import (
        NSBitmapImageRep,
        NSColor,
        NSDeviceRGBColorSpace,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSGraphicsContext,
        NSPNGFileType,
        NSRectFill,
        NSString,
    )
    from Foundation import NSMakeRect

    from jarvis.platforms.macos import MacOSPlatform

    width, height = 800, 200
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(  # noqa: E501
        None, width, height, 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0
    )
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    NSColor.whiteColor().set()
    NSRectFill(NSMakeRect(0, 0, width, height))
    attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(36),
             NSForegroundColorAttributeName: NSColor.blackColor()}  # fmt: skip
    expected = ["Server running on port 8501", "Found 12 errors in 3 files"]
    for text, y in zip(expected, (120, 40), strict=True):
        NSString.stringWithString_(text).drawAtPoint_withAttributes_((20, y), attrs)
    NSGraphicsContext.restoreGraphicsState()
    image = tmp_path / "text.png"
    rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(
        str(image), True
    )

    lines = MacOSPlatform().recognize_text(image)
    assert [line.text for line in lines] == expected
    assert all(line.confidence > 0.5 for line in lines)


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_GUI_TESTS") != "1", reason="set JARVIS_RUN_GUI_TESTS=1 to run"
)
def test_real_screen_capture_writes_a_png(tmp_path: Path) -> None:
    """Captures the real screen into a temp folder; the content is never inspected."""
    from jarvis.platforms.macos import MacOSPlatform

    platform = MacOSPlatform()
    if platform.screen_capture_allowed() is False:
        pytest.skip("Screen Recording permission not granted")
    dest = tmp_path / "screen.png"
    platform.capture_screen(dest)
    assert dest.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    dest.unlink()


@pytest.mark.gui
@pytest.mark.skipif(
    os.environ.get("JARVIS_RUN_GUI_TESTS") != "1", reason="set JARVIS_RUN_GUI_TESTS=1 to run"
)
def test_two_shortcuts_each_run_only_their_own_action() -> None:
    """Synthetic Carbon hotkey events go through the same handlers as real key presses."""
    import ctypes
    from collections.abc import Callable
    from typing import Any

    from jarvis.platforms.macos_gui import CarbonHotkeyListener, _EventHotKeyID, _fourcc
    from jarvis.runtime.hotkeys import Hotkey

    class RecordingLoop:
        def call_soon(self, fn: Callable[[], Any]) -> None:
            fn()

    carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
    carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
    carbon.CreateEvent.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_double, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]  # fmt: skip
    carbon.SetEventParameter.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_ulong, ctypes.c_void_p,
    ]  # fmt: skip
    carbon.SendEventToEventTarget.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    carbon.ReleaseEvent.argtypes = [ctypes.c_void_p]

    def press(hotkey_id: int) -> None:
        event = ctypes.c_void_p()
        assert carbon.CreateEvent(None, _fourcc("keyb"), 5, 0.0, 0, ctypes.byref(event)) == 0
        key = _EventHotKeyID(_fourcc("JRVS"), hotkey_id)
        assert (
            carbon.SetEventParameter(
                event, _fourcc("----"), _fourcc("hkid"), ctypes.sizeof(key), ctypes.byref(key)
            )
            == 0
        )
        carbon.SendEventToEventTarget(event, carbon.GetApplicationEventTarget())
        carbon.ReleaseEvent(event)

    pressed: list[str] = []
    loop: Any = RecordingLoop()
    first, second = CarbonHotkeyListener(loop), CarbonHotkeyListener(loop)
    # Unusual combinations, so a running Jarvis service (⌃⌥⌘J / ⌃⌥⌘K) isn't affected.
    first.register(Hotkey.parse("ctrl+option+cmd+shift+f11"), lambda: pressed.append("first"))
    second.register(Hotkey.parse("ctrl+option+cmd+shift+f12"), lambda: pressed.append("second"))
    try:
        press(first._id)
        press(second._id)
        press(second._id)
        press(9999)  # someone else's hotkey
        assert pressed == ["first", "second", "second"]
    finally:
        first.unregister()
        second.unregister()
