"""Path sandbox shared by the file, project and terminal tools.

Every path a tool touches is resolved (``~``, relative paths, **symlinks**) and
then checked:

* reads must stay inside ``files.read_roots`` (default: your home folder);
* writes must stay inside ``files.write_roots`` (default: Desktop, Documents,
  Downloads);
* ``files.blocked_paths`` (default includes ``~/Library``, which holds Jarvis's
  own config, audit log and models, and your app data) and hidden files or
  folders (``.ssh``, ``.env``, ``.git`` ...) are refused unless allowed;
* key and credential file names (``*.pem``, ``id_rsa`` ...) are always refused.

Paths come from the model, so this module is a security boundary. Every
decision raises :class:`~jarvis.errors.PolicyDeniedError` with a reason.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from jarvis.errors import PolicyDeniedError, ToolInputError

DEFAULT_BLOCKED_NAMES = (
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keychain",
    "*.keychain-db",
    "*.kdbx",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
)


def display(path: Path, home: Path | None = None) -> str:
    """``/Users/me/Desktop/x`` -> ``~/Desktop/x`` for messages."""
    home = home or Path.home()
    try:
        return "~/" + str(path.relative_to(home)) if path != home else "~"
    except ValueError:
        return str(path)


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def name_tokens(name: str) -> list[str]:
    """``"StockPrice-predictor_v2"`` -> ``["stock", "price", "predictor", "v2"]``."""
    spaced = _CAMEL.sub(" ", name)
    return [t for t in re.split(r"[^0-9A-Za-z]+", spaced.lower()) if t]


def match_score(query: str, name: str) -> int:
    """0 = no match. Higher is better: exact > all words > all words as substrings."""
    q = name_tokens(query)
    if not q:
        return 0
    n = name_tokens(Path(name).stem) or name_tokens(name)
    if q == n:
        return 3
    if all(any(t.startswith(w) for t in n) for w in q):
        return 2
    compact = "".join(n)
    if all(w in compact for w in q):
        return 1
    return 0


class PathPolicy:
    """Resolves and authorises paths for reading and writing."""

    def __init__(
        self,
        read_roots: Sequence[Path],
        write_roots: Sequence[Path],
        blocked_paths: Sequence[Path] = (),
        blocked_names: Iterable[str] = DEFAULT_BLOCKED_NAMES,
        *,
        allow_hidden: bool = False,
        home: Path | None = None,
    ) -> None:
        self.home = (home or Path.home()).resolve()
        self.read_roots = [self._absolute(p) for p in read_roots]
        self.write_roots = [self._absolute(p) for p in write_roots]
        self.blocked_paths = [self._absolute(p) for p in blocked_paths]
        self.blocked_names = tuple(blocked_names)
        self.allow_hidden = allow_hidden

    # -- resolution ------------------------------------------------------------------------

    def _absolute(self, path: Path) -> Path:
        path = path.expanduser()
        if not path.is_absolute():
            path = self.home / path
        return path.resolve()

    def resolve(self, raw: str) -> Path:
        """Turn a user/model supplied path into an absolute, symlink-free path."""
        text = raw.strip()
        if not text or "\x00" in text:
            raise ToolInputError("Path must be a non-empty string.")
        if text.startswith("~") and text != "~" and not text.startswith("~/"):
            # "~otheruser/..." would leave this user's home.
            raise ToolInputError(f"Unsupported path '{raw}' (use ~/... for your home folder).")
        return self._absolute(Path(text))

    # -- checks ------------------------------------------------------------------------------

    def _within(self, path: Path, roots: Sequence[Path]) -> bool:
        return any(path == root or path.is_relative_to(root) for root in roots)

    def _check_common(self, path: Path) -> None:
        for blocked in self.blocked_paths:
            if path == blocked or path.is_relative_to(blocked):
                raise PolicyDeniedError(
                    f"Not allowed: {display(path, self.home)} is in a protected location "
                    f"({display(blocked, self.home)}).",
                    hint="Protected locations are set by [files] blocked_paths.",
                )
        if any(fnmatch.fnmatch(path.name.lower(), pattern) for pattern in self.blocked_names):
            raise PolicyDeniedError(
                f"Not allowed: {path.name} looks like a key or credential file.",
                hint="Jarvis never reads or changes key files.",
            )
        if not self.allow_hidden:
            base = next((r for r in self.read_roots if path.is_relative_to(r)), None)
            parts = path.relative_to(base).parts if base else path.parts
            if any(part.startswith(".") for part in parts):
                raise PolicyDeniedError(
                    f"Not allowed: {display(path, self.home)} is (inside) a hidden file or folder.",
                    hint="Hidden files often hold settings and secrets; set [files] "
                    "allow_hidden = true to permit them.",
                )

    def check_read(self, path: Path) -> Path:
        if not self._within(path, self.read_roots):
            raise PolicyDeniedError(
                f"Not allowed: {display(path, self.home)} is outside the folders Jarvis may read.",
                hint="Readable folders are set by [files] read_roots.",
            )
        self._check_common(path)
        return path

    def check_write(self, path: Path) -> Path:
        self.check_read(path)
        if not self._within(path, self.write_roots):
            roots = ", ".join(display(r, self.home) for r in self.write_roots)
            raise PolicyDeniedError(
                f"Not allowed: Jarvis may only change files in {roots}.",
                hint="Writable folders are set by [files] write_roots.",
            )
        return path

    def readable(self, raw: str) -> Path:
        return self.check_read(self.resolve(raw))

    def writable(self, raw: str) -> Path:
        return self.check_write(self.resolve(raw))

    def is_readable(self, path: Path) -> bool:
        try:
            self.check_read(path.resolve())
        except PolicyDeniedError:
            return False
        return True
