"""Hermetic tests for the macOS adapter (fake bundles, fake subprocess, fake AppKit).

These run on any OS because PyObjC is injected rather than imported.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.errors import ToolExecutionError
from jarvis.platforms.base import AppInfo
from jarvis.platforms.macos import OPEN_BIN, MacOSAppControl, MacOSPlatform, read_app_bundle


def make_bundle(
    root: Path, name: str, info: dict[str, Any] | None = None, *, binary: bool = False
) -> Path:
    bundle = root / f"{name}.app"
    (bundle / "Contents").mkdir(parents=True)
    if info is not None:
        fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
        (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info, fmt=fmt))
    return bundle


def test_read_bundle_metadata(tmp_path: Path) -> None:
    bundle = make_bundle(
        tmp_path,
        "Visual Studio Code",
        {"CFBundleIdentifier": "com.microsoft.VSCode", "CFBundleName": "Code"},
        binary=True,
    )
    app = read_app_bundle(bundle)
    assert app == AppInfo(
        name="Visual Studio Code",
        path=bundle,
        bundle_id="com.microsoft.VSCode",
        alt_names=("Code",),
    )


@pytest.mark.parametrize(
    "plist_bytes",
    [
        None,  # no Info.plist (e.g. some game/web-app shims)
        b"<?xml version='1.0'?><plist><dict><key>x</key>\x01</dict></plist>",  # ExpatError
        b"not a plist at all",
        plistlib.dumps(["a", "list"]),  # valid plist, wrong shape
    ],
)
def test_broken_info_plist_falls_back_to_folder_name(
    tmp_path: Path, plist_bytes: bytes | None
) -> None:
    bundle = make_bundle(tmp_path, "Broken Game")
    if plist_bytes is not None:
        (bundle / "Contents" / "Info.plist").write_bytes(plist_bytes)
    app = read_app_bundle(bundle)
    assert app.name == "Broken Game"
    assert app.bundle_id is None


def test_discover_layout(tmp_path: Path) -> None:
    apps_dir = tmp_path / "Applications"
    user_dir = tmp_path / "UserApplications"
    make_bundle(apps_dir, "Safari", {"CFBundleIdentifier": "com.apple.Safari"})
    make_bundle(apps_dir / "Vendor Suite", "Editor", {"CFBundleIdentifier": "com.v.editor"})
    make_bundle(apps_dir / "Vendor Suite" / "Deeper", "TooDeep", {"CFBundleIdentifier": "x.y"})
    make_bundle(apps_dir / ".hidden", "Hidden", {"CFBundleIdentifier": "h.h"})
    inner = make_bundle(apps_dir, "Host", {"CFBundleIdentifier": "com.host"})
    make_bundle(inner / "Contents" / "Helpers", "Helper", {"CFBundleIdentifier": "com.helper"})
    (apps_dir / "README.txt").write_text("not an app")
    # Same bundle ID in a later search path: first one wins.
    make_bundle(user_dir, "Safari Copy", {"CFBundleIdentifier": "com.apple.Safari"})
    make_bundle(user_dir, "Mine", {"CFBundleIdentifier": "me.mine"})

    found = MacOSAppControl().discover([apps_dir, user_dir, tmp_path / "missing"])
    names = sorted(a.name for a in found)
    assert names == ["Editor", "Host", "Mine", "Safari"]


def test_discover_dedupes_symlinked_bundles(tmp_path: Path) -> None:
    real = make_bundle(tmp_path / "real", "Tool", None)
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    (link_dir / "Tool.app").symlink_to(real, target_is_directory=True)
    found = MacOSAppControl().discover([tmp_path / "real", link_dir])
    assert len(found) == 1


class FakeRunner:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.exc: Exception | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        assert kwargs.get("timeout"), "subprocess calls must have a timeout"
        assert "shell" not in kwargs, "subprocess calls must not use a shell"
        if self.exc:
            raise self.exc
        return self.result


APP = AppInfo("Safari", Path("/Applications/Safari.app"), "com.apple.Safari")


def test_launch_uses_open_with_bundle_path() -> None:
    runner = FakeRunner()
    MacOSAppControl(runner=runner).launch(APP)
    assert runner.calls == [[OPEN_BIN, "-a", "/Applications/Safari.app"]]


def test_launch_failure_reports_stderr() -> None:
    runner = FakeRunner(returncode=1, stderr="LSOpenURLsWithRole() failed with error -10810")
    with pytest.raises(ToolExecutionError, match="-10810") as info:
        MacOSAppControl(runner=runner).launch(APP)
    assert info.value.hint and "Gatekeeper" in info.value.hint


def test_launch_timeout() -> None:
    runner = FakeRunner()
    runner.exc = subprocess.TimeoutExpired(cmd="open", timeout=15)
    with pytest.raises(ToolExecutionError, match="Timed out"):
        MacOSAppControl(runner=runner).launch(APP)


class FakeRunningApp:
    def __init__(self, accepts_quit: bool = True, terminated: bool = False) -> None:
        self.accepts_quit = accepts_quit
        self.terminated = terminated
        self.terminate_calls = 0

    def isTerminated(self) -> bool:  # mirrors the AppKit selector name
        return self.terminated

    def terminate(self) -> bool:
        self.terminate_calls += 1
        return self.accepts_quit


def fake_appkit(instances: list[FakeRunningApp]) -> Any:
    queried: list[str] = []

    def running_with_bundle_id(bundle_id: str) -> list[FakeRunningApp]:
        queried.append(bundle_id)
        return instances

    return SimpleNamespace(
        NSRunningApplication=SimpleNamespace(
            runningApplicationsWithBundleIdentifier_=running_with_bundle_id
        ),
        queried=queried,
    )


def test_is_running_queries_by_bundle_id() -> None:
    appkit = fake_appkit([FakeRunningApp()])
    control = MacOSAppControl(appkit_loader=lambda: appkit)
    assert control.is_running(APP) is True
    assert appkit.queried == ["com.apple.Safari"]


def test_terminated_instances_do_not_count() -> None:
    control = MacOSAppControl(appkit_loader=lambda: fake_appkit([FakeRunningApp(terminated=True)]))
    assert control.is_running(APP) is False


def test_is_running_unknown_without_bundle_id() -> None:
    control = MacOSAppControl(appkit_loader=lambda: pytest.fail("AppKit must not be loaded"))
    assert control.is_running(AppInfo("X", Path("/Applications/X.app"))) is None


def test_quit_terminates_every_instance() -> None:
    instances = [FakeRunningApp(), FakeRunningApp()]
    MacOSAppControl(appkit_loader=lambda: fake_appkit(instances)).quit(APP)
    assert [i.terminate_calls for i in instances] == [1, 1]


def test_quit_refused() -> None:
    control = MacOSAppControl(appkit_loader=lambda: fake_appkit([FakeRunningApp(False)]))
    with pytest.raises(ToolExecutionError, match="did not accept"):
        control.quit(APP)


def test_quit_without_bundle_id_is_an_error() -> None:
    with pytest.raises(ToolExecutionError, match="no identifier"):
        MacOSAppControl().quit(AppInfo("X", Path("/Applications/X.app")))


def test_diagnostics_detect_rosetta() -> None:
    runner = FakeRunner(stdout="1\n")
    platform = MacOSPlatform(
        apps=MacOSAppControl(runner=runner, appkit_loader=lambda: object()), runner=runner
    )
    checks = {c.name: c for c in platform.diagnostics()}
    assert checks["Native arm64 Python"].ok is False
    assert "Rosetta" in checks["Native arm64 Python"].detail
    assert checks["PyObjC AppKit"].ok is True
