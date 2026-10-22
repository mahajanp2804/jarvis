"""Per-user locations for configuration and logs.

Only ``sys.platform`` checks live here (no OS-specific imports), so this module
is safe to import on every platform.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Jarvis"


def config_dir() -> Path:
    """Directory holding ``config.toml`` and the SQLite memory database."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return (Path(base) if base else home / "AppData" / "Roaming") / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else home / ".config") / APP_NAME.lower()


def default_log_dir() -> Path:
    """Directory for the rotating application log and the audit log."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / APP_NAME
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_NAME / "Logs"
    base = os.environ.get("XDG_STATE_HOME")
    return (Path(base) if base else home / ".local" / "state") / APP_NAME.lower() / "logs"


def default_config_file() -> Path:
    """Default path of the user's configuration file."""
    return config_dir() / "config.toml"


def runtime_dir() -> Path:
    """Directory for the service's control socket and instance lock (0700)."""
    return config_dir() / "run"


def default_models_dir() -> Path:
    """Directory for local model files (multi-GB; kept out of the source tree).

    On macOS this is under Application Support, which is neither inside the git
    checkout nor in an iCloud-synced Desktop/Documents folder. On Windows it is
    under %LOCALAPPDATA% so the files never enter a roaming profile.
    """
    home = Path.home()
    if sys.platform == "darwin":
        return config_dir() / "models"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_NAME / "models"
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else home / ".local" / "share") / APP_NAME.lower() / "models"


def default_prompt_cache_dir() -> Path:
    """Directory for saved llama.cpp prompt states (~250 MB each, rebuildable)."""
    home = Path.home()
    if sys.platform == "darwin":
        return config_dir() / "prompt-cache"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / APP_NAME / "prompt-cache"
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else home / ".cache") / APP_NAME.lower() / "prompt-cache"


def default_memory_path() -> Path:
    """The SQLite memory database (preferences, task history, workflows)."""
    return config_dir() / "memory.sqlite3"
