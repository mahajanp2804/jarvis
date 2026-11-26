"""The macOS Shortcuts adapter against a fake `shortcuts` program (no real shortcut runs)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from jarvis.errors import ToolExecutionError
from jarvis.platforms.macos import MacOSPlatform

FAKE_SHORTCUTS = """\
#!{python}
import pathlib, sys, time
args = sys.argv[1:]
with open({log!r}, "a") as log:
    log.write(" ".join(args) + "\\n")
if args == ["list"]:
    print("Echo\\nSilent\\nBroken\\nSlow")
    sys.exit(0)
name = args[1]
options = dict(zip(args[2::2], args[3::2]))
if name == "Echo":
    given = pathlib.Path(options["--input-path"]).read_text() if "--input-path" in options else ""
    pathlib.Path(options["--output-path"]).write_text("echo: " + given)
elif name == "Silent":
    pass
elif name == "Broken":
    print("Error: The action could not run.", file=sys.stderr)
    sys.exit(1)
elif name == "Slow":
    time.sleep(30)
else:
    print("Error: Couldn't find shortcut", file=sys.stderr)
    sys.exit(1)
"""


@pytest.fixture
def platform(tmp_path: Path) -> MacOSPlatform:
    script = tmp_path / "shortcuts"
    script.write_text(FAKE_SHORTCUTS.format(python=sys.executable, log=str(tmp_path / "log")))
    script.chmod(0o755)
    return MacOSPlatform(shortcuts_bin=str(script))


def run(platform: MacOSPlatform, name: str, text: str | None = None, timeout: float = 10) -> str:
    return platform.run_shortcut(name, text, timeout=timeout, cancel=threading.Event())


def test_lists_shortcuts(platform: MacOSPlatform) -> None:
    assert platform.list_shortcuts() == ["Echo", "Silent", "Broken", "Slow"]


def test_runs_with_input_and_returns_text_output(platform: MacOSPlatform, tmp_path: Path) -> None:
    assert run(platform, "Echo", "hello there") == "echo: hello there"
    logged = (tmp_path / "log").read_text().splitlines()[-1]
    assert logged.startswith("run Echo --output-path ")
    assert "--output-type public.plain-text" in logged and "--input-path" in logged
    assert run(platform, "Echo") == "echo: "  # no input: no --input-path
    assert "--input-path" not in (tmp_path / "log").read_text().splitlines()[-1]


def test_no_output_is_empty_text(platform: MacOSPlatform) -> None:
    assert run(platform, "Silent") == ""


def test_failures_are_reported(platform: MacOSPlatform) -> None:
    with pytest.raises(
        ToolExecutionError, match="“Broken” failed: Error: The action could not run"
    ):
        run(platform, "Broken")
    with pytest.raises(ToolExecutionError, match="Couldn't find shortcut"):
        run(platform, "Missing")


def test_slow_shortcuts_are_stopped(platform: MacOSPlatform) -> None:
    started = time.monotonic()
    with pytest.raises(ToolExecutionError, match=r"didn.t finish within 0\.5 s") as info:
        run(platform, "Slow", timeout=0.5)
    assert info.value.hint and "timeout_seconds" in info.value.hint
    assert time.monotonic() - started < 5


def test_cancel_stops_a_running_shortcut(platform: MacOSPlatform) -> None:
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(ToolExecutionError, match="Stopped the shortcut “Slow”"):
        platform.run_shortcut("Slow", None, timeout=30, cancel=cancel)
    assert time.monotonic() - started < 5


@pytest.mark.integration
@pytest.mark.skipif(not Path("/usr/bin/shortcuts").exists(), reason="needs macOS Shortcuts")
def test_real_shortcuts_list_is_readable() -> None:
    names = MacOSPlatform().list_shortcuts()  # read-only: never runs a shortcut
    assert isinstance(names, list) and all(isinstance(n, str) and n for n in names)
