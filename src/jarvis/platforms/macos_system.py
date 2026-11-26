"""Everyday system controls on macOS, with built-in command-line tools only.

* Volume and mute: ``osascript`` with fixed scripts from Standard Additions,
  which need no Automation permission. Only validated integers and booleans
  are put into a script, never text from a request.
* Wi-Fi: ``networksetup -setairportpower``. The Wi-Fi device (usually ``en0``)
  is looked up, not assumed.
* Display: ``pmset displaysleepnow``. macOS locks the screen if it asks for a
  password after sleep (System Settings > Lock Screen).
* State also reads the battery (``pmset -g batt``) and dark mode
  (``defaults read -g AppleInterfaceStyle``).

Brightness, Focus, Bluetooth and dark mode switching have no permission-free
command; use the Shortcuts bridge for them.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

from jarvis.errors import OSPermissionError, ToolExecutionError
from jarvis.platforms.base import SystemControls, SystemState

OSASCRIPT = "/usr/bin/osascript"
NETWORKSETUP = "/usr/sbin/networksetup"
PMSET = "/usr/bin/pmset"
DEFAULTS = "/usr/bin/defaults"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def parse_volume(text: str) -> tuple[int | None, bool | None]:
    """'output volume:50, input volume:27, alert volume:100, output muted:false'."""
    level = re.search(r"output volume:(\d+)", text)
    muted = re.search(r"output muted:(true|false)", text)
    return (
        int(level.group(1)) if level else None,  # "missing value" for e.g. HDMI outputs
        muted.group(1) == "true" if muted else None,
    )


def parse_battery(text: str) -> tuple[int | None, bool | None, bool | None]:
    """(percent, charging, on battery power) from ``pmset -g batt``."""
    percent = re.search(r"(\d+)%;\s*([a-z ]+);", text)
    source = re.search(r"drawing from '([^']+)'", text)
    on_battery = source.group(1) == "Battery Power" if source else None
    if not percent:
        return None, None, on_battery  # a Mac without a battery
    state = percent.group(2).strip()
    return int(percent.group(1)), state in {"charging", "charged", "finishing charge"}, on_battery


def wifi_device(listing: str) -> str | None:
    """The Wi-Fi device ("en0") from ``networksetup -listallhardwareports``."""
    match = re.search(r"Hardware Port: Wi-Fi\s*\nDevice: (\S+)", listing)
    return match.group(1) if match else None


class MacSystemControls(SystemControls):
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self._run = runner
        self._wifi: str | None = None

    def _call(self, argv: list[str], what: str) -> str:
        try:
            proc = self._run(argv, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolExecutionError(f"Couldn't {what}: {exc}") from exc
        output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or "** Error" in output:
            if "administrator" in output.lower() or "not permitted" in output.lower():
                raise OSPermissionError(
                    f"macOS didn't allow Jarvis to {what}.",
                    hint="This needs an administrator account.",
                )
            raise ToolExecutionError(f"Couldn't {what}: {output.strip() or proc.returncode}")
        return proc.stdout or ""

    def _wifi_device(self) -> str | None:
        if self._wifi is None:
            self._wifi = wifi_device(
                self._call([NETWORKSETUP, "-listallhardwareports"], "find the Wi-Fi device")
            )
        return self._wifi

    def state(self) -> SystemState:
        volume, muted = parse_volume(
            self._call([OSASCRIPT, "-e", "get volume settings"], "read the volume")
        )
        wifi_on = None
        device = self._wifi_device()
        if device:
            power = self._call([NETWORKSETUP, "-getairportpower", device], "read Wi-Fi")
            wifi_on = power.strip().endswith("On")
        percent, charging, on_battery = parse_battery(
            self._call([PMSET, "-g", "batt"], "read the battery")
        )
        try:  # fails when the setting doesn't exist, which means light mode
            dark = self._run(
                [DEFAULTS, "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=15,
            )  # fmt: skip
            dark_mode: bool | None = dark.returncode == 0 and dark.stdout.strip() == "Dark"
        except (OSError, subprocess.TimeoutExpired):
            dark_mode = None
        return SystemState(volume, muted, wifi_on, percent, charging, on_battery, dark_mode)

    def set_volume(self, level: int) -> None:
        if not 0 <= level <= 100:
            raise ValueError("volume must be 0-100")
        self._call([OSASCRIPT, "-e", f"set volume output volume {int(level)}"], "set the volume")

    def set_muted(self, muted: bool) -> None:
        value = "true" if muted else "false"
        self._call([OSASCRIPT, "-e", f"set volume output muted {value}"], "mute the sound")

    def set_wifi(self, on: bool) -> None:
        device = self._wifi_device()
        if device is None:
            raise ToolExecutionError("This Mac has no Wi-Fi device.")
        state = "on" if on else "off"
        self._call([NETWORKSETUP, "-setairportpower", device, state], f"turn Wi-Fi {state}")

    def sleep_display(self) -> None:
        self._call([PMSET, "displaysleepnow"], "turn the display off")
