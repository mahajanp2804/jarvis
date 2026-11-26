"""Everyday system controls: status, volume, Wi-Fi and display sleep.

* ``system.status`` reads the volume, Wi-Fi, battery and dark mode (READ_ONLY).
* ``system.volume`` sets a level, turns it up or down by 10, or mutes (LOW:
  easily undone, like Siri's volume commands). A level or "up" also unmutes.
* ``system.wifi`` turns Wi-Fi on (LOW) or off (MEDIUM, asks first: it cuts this
  Mac off the internet, including Jarvis's own web tools).
* ``system.sleep_display`` turns the display off (LOW). macOS locks the screen
  if it asks for a password after sleep.

Each change reads the state back, so the reply says what actually happened.
Brightness, Focus and Bluetooth have no permission-free command; they go
through ``shortcuts.run``.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import ToolExecutionError
from jarvis.platforms.base import PlatformAdapter, SystemControls, SystemState
from jarvis.tools.base import PreparedAction, ToolSpec

VOLUME_STEP = 10


class StatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VolumeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: int | None = Field(default=None, ge=0, le=100, description="0-100")
    change: Literal["up", "down"] | None = Field(default=None, description="up or down")
    mute: bool | None = Field(default=None, description="true to mute, false to unmute")

    @model_validator(mode="after")
    def _exactly_one(self) -> VolumeArgs:
        if sum(v is not None for v in (self.level, self.change, self.mute)) != 1:
            raise ValueError("give exactly one of level, change or mute")
        return self


class WifiArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    on: bool = Field(description="true = on, false = off")


def describe_state(state: SystemState) -> str:
    parts = []
    if state.volume is not None:
        parts.append(f"Volume {state.volume}%" + (" (muted)" if state.muted else ""))
    elif state.muted is not None:
        parts.append("Sound muted" if state.muted else "Sound on")
    if state.wifi_on is not None:
        parts.append(f"Wi-Fi {'on' if state.wifi_on else 'off'}")
    if state.battery_percent is not None:
        how = "charging" if state.charging else "on battery" if state.on_battery_power else ""
        parts.append(f"Battery {state.battery_percent}%" + (f", {how}" if how else ""))
    if state.dark_mode is not None:
        parts.append("Dark mode" if state.dark_mode else "Light mode")
    return " · ".join(parts) or "Couldn't read the system state."


class SystemTools:
    """Builds the ``system.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    def _controls(self) -> SystemControls:
        return self._platform.system_controls()  # UnsupportedPlatformError off macOS

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "system.status",
                "Show the Mac's volume, Wi-Fi, battery and dark mode.",
                StatusArgs,
                RiskLevel.READ_ONLY,
                self._prepare_status,
                idempotent=True,
            ),
            ToolSpec(
                "system.volume",
                "Change the speaker volume: set a level, turn it up or down, or mute. "
                "Give one argument.",
                VolumeArgs,
                RiskLevel.LOW,
                self._prepare_volume,
                idempotent=True,
            ),
            ToolSpec(
                "system.wifi",
                "Turn Wi-Fi on or off.",
                WifiArgs,
                RiskLevel.LOW,
                self._prepare_wifi,
                idempotent=True,
            ),
            ToolSpec(
                "system.sleep_display",
                "Turn the display off (sleep the screen).",
                StatusArgs,
                RiskLevel.LOW,
                self._prepare_sleep_display,
            ),
        ]

    def _prepare_status(self, args: StatusArgs) -> PreparedAction:
        controls = self._controls()

        def run(cancel: threading.Event) -> ToolResult:
            state = controls.state()
            return ToolResult(
                tool="system.status",
                status=ToolStatus.SUCCESS,
                message=describe_state(state),
                data=asdict(state),
            )

        return PreparedAction(description="Read the volume, Wi-Fi and battery", run=run)

    def _prepare_volume(self, args: VolumeArgs) -> PreparedAction:
        controls = self._controls()
        before = controls.state()
        if args.mute is not None:
            noop = before.muted is args.mute
            description = "Mute the sound" if args.mute else "Unmute the sound"
        else:
            if before.volume is None:
                raise ToolExecutionError(
                    "The current sound output's volume can't be set from the Mac.",
                    hint="Use the speaker's or display's own volume buttons.",
                )
            if args.level is not None:
                target = args.level
            elif args.change == "up":
                target = min(100, before.volume + VOLUME_STEP)
            else:
                target = max(0, before.volume - VOLUME_STEP)
            unmute = bool(before.muted) and args.change != "down"
            noop = target == before.volume and not unmute
            description = f"Set the volume to {target}%" + (" and unmute" if unmute else "")

        def run(cancel: threading.Event) -> ToolResult:
            after = before
            if not noop:
                if args.mute is not None:
                    controls.set_muted(args.mute)
                else:
                    controls.set_volume(target)
                    if unmute:
                        controls.set_muted(False)
                after = controls.state()  # report what macOS actually did
            now = "already" if noop else "now"
            if args.mute is not None:
                message = f"The sound is {now} {'muted' if after.muted else 'on'}."
            else:
                was = "" if noop else f" (was {before.volume}%)"
                message = f"The volume is {now} {after.volume}%{was}."
            return ToolResult(
                tool="system.volume",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"volume": after.volume, "muted": after.muted},
            )

        return PreparedAction(description=description, run=run, noop=noop)

    def _prepare_wifi(self, args: WifiArgs) -> PreparedAction:
        controls = self._controls()
        before = controls.state()
        if before.wifi_on is None:
            raise ToolExecutionError("This Mac has no Wi-Fi.")
        word = "on" if args.on else "off"
        noop = before.wifi_on is args.on

        def run(cancel: threading.Event) -> ToolResult:
            if not noop:
                controls.set_wifi(args.on)
                if controls.state().wifi_on is not args.on:
                    raise ToolExecutionError(f"macOS accepted the request, but Wi-Fi isn't {word}.")
            return ToolResult(
                tool="system.wifi",
                status=ToolStatus.SUCCESS,
                message=f"Wi-Fi is already {word}." if noop else f"Turned Wi-Fi {word}.",
                data={"wifi_on": args.on},
            )

        return PreparedAction(
            description=f"Turn Wi-Fi {word}",
            run=run,
            noop=noop,
            warning=None
            if args.on
            else "This Mac goes offline, including Jarvis's web tools, until Wi-Fi is back on.",
            risk=None if args.on else RiskLevel.MEDIUM,
        )

    def _prepare_sleep_display(self, args: StatusArgs) -> PreparedAction:
        controls = self._controls()

        def run(cancel: threading.Event) -> ToolResult:
            controls.sleep_display()
            return ToolResult(
                tool="system.sleep_display",
                status=ToolStatus.SUCCESS,
                message="Turned the display off.",
            )

        return PreparedAction(description="Turn the display off", run=run)
