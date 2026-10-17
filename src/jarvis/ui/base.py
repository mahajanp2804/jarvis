"""Minimal user-interface contract for the background service.

Three blocking dialogs are enough for the activation flow:

* ``prompt``: ask for a request (or an answer to a clarifying question);
* ``confirm``: approve one specific action (default is **No**);
* ``show``: display a reply.

All methods run on the event-loop (main) thread. Implementations: native
dialogs on macOS (:mod:`jarvis.platforms.macos_gui`) and :class:`HeadlessUI`
for running without a display.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class UserInterface(ABC):
    @abstractmethod
    def prompt(self, title: str, message: str) -> str | None:
        """Ask for text. ``None`` if the user cancelled."""

    @abstractmethod
    def confirm(self, title: str, message: str) -> bool:
        """Ask to approve one action. Only an explicit approval returns True."""

    @abstractmethod
    def show(self, title: str, message: str) -> None:
        """Display a reply."""


class HeadlessUI(UserInterface):
    """No display: nothing to ask, confirmations denied, replies logged."""

    def prompt(self, title: str, message: str) -> str | None:
        return None

    def confirm(self, title: str, message: str) -> bool:
        log.info("Confirmation denied (headless): %s", message)
        return False

    def show(self, title: str, message: str) -> None:
        log.info("%s: %s", title, message)
