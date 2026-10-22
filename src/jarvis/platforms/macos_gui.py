"""macOS GUI pieces for the background service: run loop, global hotkey, dialogs.

* :class:`AppKitEventLoop` runs ``NSApplication`` as an *accessory* app (no Dock
  icon, no menu bar). Idle, it sleeps in the run loop and uses no CPU.
* :class:`CarbonHotkeyListener` uses Carbon's ``RegisterEventHotKey``
  through ctypes. Only the one registered key combination is delivered, so it
  needs **no Accessibility or Input Monitoring permission** and never sees other
  keystrokes (unlike event taps such as pynput's).
* :class:`AlertUI` uses standard ``NSAlert`` dialogs. In confirmation dialogs,
  Cancel is the default button, so pressing Return never approves anything.

Everything here must run on the main thread; :mod:`jarvis.runtime.event_loop`
marshals calls from worker threads.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import threading
from collections.abc import Callable
from typing import Any

from jarvis.runtime.event_loop import EventLoop
from jarvis.runtime.hotkeys import Hotkey, HotkeyError, HotkeyListener
from jarvis.ui.base import UserInterface

log = logging.getLogger(__name__)

# -- run loop --------------------------------------------------------------------


class AppKitEventLoop(EventLoop):
    """``NSApplication`` run loop in accessory mode."""

    def __init__(self) -> None:
        super().__init__()
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        self._app = NSApplication.sharedApplication()
        self._app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    def run(self) -> None:
        if not self.stopping:
            self._app.run()

    def call_soon(self, fn: Callable[[], Any]) -> None:
        from PyObjCTools import AppHelper

        # performSelectorOnMainThread in the common modes: also runs while a
        # modal dialog is open, so shutdown is never blocked by a dialog.
        AppHelper.callAfter(fn)

    def _wake_for_stop(self) -> None:
        self.call_soon(self._stop_on_main_thread)

    def _stop_on_main_thread(self) -> None:
        if self._app.modalWindow() is not None:
            # NSApp.stop_ during a modal session would only end the dialog, so
            # end the dialog first and stop the main loop on the next turn.
            log.info("Closing an open dialog for shutdown")
            self._app.abortModal()
            self.call_soon(self._stop_on_main_thread)
            return
        self._app.stop_(None)
        self._post_wakeup_event()  # stop_ takes effect after the next event

    def _post_wakeup_event(self) -> None:
        from AppKit import NSApplicationDefined, NSEvent
        from Foundation import NSMakePoint

        event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(  # noqa: E501
            NSApplicationDefined, NSMakePoint(0, 0), 0, 0, 0, None, 0, 0, 0
        )
        self._app.postEvent_atStart_(event, True)


# -- global hotkey (Carbon) ----------------------------------------------------------

_KEY_CODES = {  # kVK_* virtual key codes (ANSI layout positions)
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31,
    "u": 32, "i": 34, "p": 35, "return": 36, "l": 37, "j": 38, "k": 40, "n": 45,
    "m": 46, "tab": 48, "space": 49, "escape": 53, "f1": 122, "f2": 120, "f3": 99,
    "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
    "f11": 103, "f12": 111,
}  # fmt: skip
_MODIFIER_FLAGS = {"cmd": 1 << 8, "shift": 1 << 9, "option": 1 << 11, "ctrl": 1 << 12}
_EVENT_HOTKEY_EXISTS = -9878
_EVENT_NOT_HANDLED = -9874
_HOTKEY_IDS = itertools.count(1)  # one id per listener, so each handles only its own key


def carbon_codes(hotkey: Hotkey) -> tuple[int, int]:
    """(virtual key code, Carbon modifier mask) for ``hotkey``."""
    mask = 0
    for modifier in hotkey.modifiers:
        mask |= _MODIFIER_FLAGS[modifier]
    return _KEY_CODES[hotkey.key], mask


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class CarbonHotkeyListener(HotkeyListener):
    """Global shortcut via ``RegisterEventHotKey`` (no Accessibility permission needed)."""

    def __init__(self, loop: EventLoop) -> None:
        self._loop = loop
        self._carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
        self._carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        self._carbon.InstallEventHandler.argtypes = [
            ctypes.c_void_p,
            _HANDLER,
            ctypes.c_ulong,
            ctypes.POINTER(_EventTypeSpec),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._carbon.RegisterEventHotKey.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            _EventHotKeyID,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._carbon.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        self._carbon.RemoveEventHandler.argtypes = [ctypes.c_void_p]
        self._carbon.GetEventParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._id = next(_HOTKEY_IDS)
        self._hotkey_ref: ctypes.c_void_p | None = None
        self._handler_ref: ctypes.c_void_p | None = None
        self._c_callback: Any = None  # keep a reference so ctypes doesn't free it

    def register(self, hotkey: Hotkey, callback: Callable[[], None]) -> None:
        self.unregister()

        def on_event(_call_ref: int, event: int, _user: int) -> int:
            # Every listener's handler sees every hotkey press: act only on our own id
            # and let the other handlers have the rest.
            pressed = _EventHotKeyID()
            status = self._carbon.GetEventParameter(
                event, _fourcc("----"), _fourcc("hkid"), None,
                ctypes.sizeof(pressed), None, ctypes.byref(pressed),
            )  # fmt: skip
            if status != 0 or pressed.id != self._id:
                return _EVENT_NOT_HANDLED
            # Leave the Carbon handler quickly; open dialogs from the run loop.
            self._loop.call_soon(callback)
            return 0

        self._c_callback = _HANDLER(on_event)
        target = self._carbon.GetApplicationEventTarget()
        spec = _EventTypeSpec(_fourcc("keyb"), 5)  # kEventClassKeyboard, kEventHotKeyPressed
        handler_ref = ctypes.c_void_p()
        status = self._carbon.InstallEventHandler(
            target, self._c_callback, 1, ctypes.byref(spec), None, ctypes.byref(handler_ref)
        )
        if status != 0:
            raise HotkeyError(f"Could not install the hotkey handler (OSStatus {status}).")
        self._handler_ref = handler_ref

        key_code, modifiers = carbon_codes(hotkey)
        hotkey_ref = ctypes.c_void_p()
        status = self._carbon.RegisterEventHotKey(
            key_code, modifiers, _EventHotKeyID(_fourcc("JRVS"), self._id), target, 0,
            ctypes.byref(hotkey_ref),
        )  # fmt: skip
        if status != 0:
            self.unregister()
            if status == _EVENT_HOTKEY_EXISTS:
                raise HotkeyError(
                    f"The shortcut {hotkey.symbol()} is already registered.",
                    hint="Choose another [service] hotkey in config.toml.",
                )
            raise HotkeyError(f"macOS refused the shortcut {hotkey.symbol()} (OSStatus {status}).")
        self._hotkey_ref = hotkey_ref
        log.info("Registered global shortcut %s", hotkey.symbol())

    def unregister(self) -> None:
        if self._hotkey_ref is not None:
            self._carbon.UnregisterEventHotKey(self._hotkey_ref)
            self._hotkey_ref = None
        if self._handler_ref is not None:
            self._carbon.RemoveEventHandler(self._handler_ref)
            self._handler_ref = None


# -- dialogs ------------------------------------------------------------------------

_MAX_TEXT = 2000


class AlertUI(UserInterface):
    """Native ``NSAlert`` dialogs."""

    def __init__(self, auto_dismiss_seconds: float = 30.0) -> None:
        self._auto_dismiss = auto_dismiss_seconds

    def _activate(self) -> None:
        from AppKit import NSApp

        # We are responding to the user's shortcut, so taking focus is expected.
        NSApp.activateIgnoringOtherApps_(True)

    def prompt(self, title: str, message: str) -> str | None:
        return self._text_dialog(title, message, "")

    def edit(self, title: str, message: str, text: str) -> str | None:
        return self._text_dialog(title, message, text)

    def _text_dialog(self, title: str, message: str, text: str) -> str | None:
        from AppKit import NSAlert, NSAlertFirstButtonReturn, NSTextField
        from Foundation import NSMakeRect

        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(_clip(message))
        field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 380, 24))
        field.setPlaceholderString_('e.g. "open my browser and the notes app"')
        field.setStringValue_(_clip(text))
        alert.setAccessoryView_(field)
        alert.addButtonWithTitle_("Run")
        alert.addButtonWithTitle_("Cancel")
        alert.window().setInitialFirstResponder_(field)
        self._activate()
        response = alert.runModal()
        text = str(field.stringValue()).strip()
        return text if response == NSAlertFirstButtonReturn and text else None

    def confirm(self, title: str, message: str) -> bool:
        from AppKit import NSAlert, NSAlertSecondButtonReturn, NSAlertStyleWarning

        alert = NSAlert.alloc().init()
        alert.setAlertStyle_(NSAlertStyleWarning)
        alert.setMessageText_(title)
        alert.setInformativeText_(_clip(message))
        alert.addButtonWithTitle_("Cancel")  # first button = default (Return)
        alert.addButtonWithTitle_("Allow")
        self._activate()
        return bool(alert.runModal() == NSAlertSecondButtonReturn)

    def listen(self, title: str, message: str, finished: threading.Event) -> bool:
        from AppKit import NSAlert, NSAlertFirstButtonReturn, NSApp, NSModalPanelRunLoopMode
        from Foundation import NSRunLoop, NSTimer

        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(_clip(message))
        alert.addButtonWithTitle_("Done")
        alert.addButtonWithTitle_("Cancel")

        def close_when_finished(_timer: Any) -> None:
            if finished.is_set() and NSApp.modalWindow() is not None:
                NSApp.stopModalWithCode_(NSAlertFirstButtonReturn)

        # Checked 10 times a second, only while this dialog is open.
        timer = NSTimer.timerWithTimeInterval_repeats_block_(0.1, True, close_when_finished)
        NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSModalPanelRunLoopMode)
        self._activate()
        try:
            return bool(alert.runModal() == NSAlertFirstButtonReturn)
        finally:
            timer.invalidate()

    def show(self, title: str, message: str) -> None:
        from AppKit import NSAlert, NSApp, NSModalPanelRunLoopMode
        from Foundation import NSRunLoop, NSTimer

        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(_clip(message))
        alert.addButtonWithTitle_("OK")
        timer = None
        if self._auto_dismiss > 0:
            # Dismiss automatically so an unattended reply doesn't sit on screen.
            timer = NSTimer.timerWithTimeInterval_repeats_block_(
                self._auto_dismiss, False, lambda _t: NSApp.abortModal()
            )
            NSRunLoop.currentRunLoop().addTimer_forMode_(timer, NSModalPanelRunLoopMode)
        self._activate()
        alert.runModal()
        if timer is not None:
            timer.invalidate()


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"
