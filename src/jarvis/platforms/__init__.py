"""Operating-system adapters. Use :func:`get_platform` instead of importing one directly."""

from __future__ import annotations

import sys

from jarvis.errors import UnsupportedPlatformError
from jarvis.platforms.base import PlatformAdapter


def get_platform() -> PlatformAdapter:
    """Return the adapter for the current OS (OS-specific modules are imported lazily)."""
    if sys.platform == "darwin":
        from jarvis.platforms.macos import MacOSPlatform

        return MacOSPlatform()
    if sys.platform == "win32":
        from jarvis.platforms.windows import WindowsPlatform

        return WindowsPlatform()
    raise UnsupportedPlatformError(
        f"Platform '{sys.platform}' is not supported.",
        hint="Jarvis currently supports macOS; Windows is planned.",
    )
