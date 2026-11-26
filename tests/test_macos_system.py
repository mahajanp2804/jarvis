"""macOS system controls with a fake command runner (nothing on this Mac changes)."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from jarvis.errors import OSPermissionError, ToolExecutionError
from jarvis.platforms.macos_system import (
    MacSystemControls,
    parse_battery,
    parse_volume,
    wifi_device,
)

PORTS = """\
Hardware Port: Ethernet Adapter (en4)
Device: en4
Ethernet Address: 00:00:00:00:00:01

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: 00:00:00:00:00:02
"""


VOLUME = "output volume:50, input volume:27, alert volume:100, output muted:false"
BATTERY = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=35782755)\t80%; charging; (no estimate) present: true\n"
)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.replies: dict[str, tuple[int, str, str]] = {
            "get volume settings": (0, VOLUME, ""),
            "-listallhardwareports": (0, PORTS, ""),
            "-getairportpower": (0, "Wi-Fi Power (en0): On\n", ""),
            "batt": (0, BATTERY, ""),
            "AppleInterfaceStyle": (0, "Dark\n", ""),
        }

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") and "shell" not in kwargs
        self.calls.append(argv)
        for key, (code, out, err) in self.replies.items():
            if key in argv or key in " ".join(argv):
                return subprocess.CompletedProcess(argv, code, out, err)
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


def test_parsers() -> None:
    assert parse_volume(
        "output volume:50, input volume:27, alert volume:100, output muted:true"
    ) == (50, True)
    assert parse_volume("output volume:missing value, output muted:missing value") == (None, None)
    assert parse_battery(
        "Now drawing from 'Battery Power'\n -InternalBattery-0\t41%; discharging; 3:12 remaining"
    ) == (41, False, True)
    assert parse_battery(
        "Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged; 0:00"
    ) == (100, True, False)
    assert parse_battery("Now drawing from 'AC Power'\n") == (None, None, False)  # desktop Mac
    assert wifi_device(PORTS) == "en0"
    assert wifi_device("Hardware Port: Ethernet\nDevice: en4\n") is None


def test_state(runner: FakeRunner) -> None:
    state = MacSystemControls(runner).state()
    assert (state.volume, state.muted, state.wifi_on) == (50, False, True)
    assert (state.battery_percent, state.charging, state.on_battery_power) == (80, True, False)
    assert state.dark_mode is True
    runner.replies["AppleInterfaceStyle"] = (1, "", "The domain/default pair does not exist")
    assert MacSystemControls(runner).state().dark_mode is False  # light mode


def test_changes_use_fixed_commands(runner: FakeRunner) -> None:
    controls = MacSystemControls(runner)
    controls.set_volume(35)
    controls.set_muted(True)
    controls.set_wifi(False)
    controls.sleep_display()
    assert ["/usr/bin/osascript", "-e", "set volume output volume 35"] in runner.calls
    assert ["/usr/bin/osascript", "-e", "set volume output muted true"] in runner.calls
    assert ["/usr/sbin/networksetup", "-setairportpower", "en0", "off"] in runner.calls
    assert ["/usr/bin/pmset", "displaysleepnow"] in runner.calls
    with pytest.raises(ValueError):
        controls.set_volume(150)


def test_errors(runner: FakeRunner) -> None:
    runner.replies["-setairportpower"] = (
        1, "", "You need administrator privileges to turn Wi-Fi off.",
    )  # fmt: skip
    with pytest.raises(OSPermissionError, match="didn't allow Jarvis to turn Wi-Fi off"):
        MacSystemControls(runner).set_wifi(False)
    runner.replies["-listallhardwareports"] = (0, "Hardware Port: Ethernet\nDevice: en4\n", "")
    with pytest.raises(ToolExecutionError, match="no Wi-Fi device"):
        MacSystemControls(runner).set_wifi(True)


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_real_state_is_readable() -> None:
    state = MacSystemControls().state()  # read-only: changes nothing
    assert state.volume is None or 0 <= state.volume <= 100
    assert state.dark_mode in (True, False)
