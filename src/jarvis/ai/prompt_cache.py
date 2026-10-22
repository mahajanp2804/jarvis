"""On-disk cache of llama.cpp's state for the system prompt.

A model load starts with an empty KV cache, so the first request after every
load re-reads the whole system prompt (~1,750 tokens, ~5.5 s on an M4). That
prompt only changes when the tools, the installed-app list or the settings do,
so its state is saved once and restored after later loads (~0.06 s).

* Entries are keyed by a hash of the system prompt, the model file (name, size,
  modification time), the context length, the chat format and the
  llama-cpp-python version. Any change gives a new key; old entries age out.
* Only the system prompt is stored, never a request.
* Files are written under a temporary name and renamed, and carry a CRC32 in
  their name: llama.cpp rejects truncated files but would load flipped bytes.
* The directory is capped in bytes; least recently used entries go first.

This module only manages files. :mod:`jarvis.ai.llama_cpp_backend` does the
llama.cpp calls.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

SUFFIX = ".llstate"
_TMP_MARKER = ".tmp-"
_STALE_TMP_SECONDS = 600.0
_FORMAT = 1  # bump when the file layout or key changes


def file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb", buffering=0) as f:
        while chunk := f.read(8 << 20):
            crc = zlib.crc32(chunk, crc)
    return crc


def system_prefix_length(
    tokens: Sequence[int], system_prompt: str, detokenize: Callable[[Sequence[int]], bytes]
) -> int:
    """Number of leading ``tokens`` that end inside ``system_prompt``'s text.

    ``tokens`` is a whole chat prompt as the model saw it. The result covers the
    template header and the system prompt but none of the request, so the state
    for it is shared by every request. 0 if the system prompt isn't found.
    """
    full = detokenize(tokens)
    marker = system_prompt.encode("utf-8")
    start = full.find(marker)
    if start < 0 or not marker:
        return 0
    end = start + len(marker)
    lo, hi = 0, len(tokens)
    while lo < hi:  # largest n whose text still ends inside the system prompt
        mid = (lo + hi + 1) // 2
        if len(detokenize(tokens[:mid])) <= end:
            lo = mid
        else:
            hi = mid - 1
    return lo


class PromptCache:
    """A directory of saved prompt states, at most ``max_bytes`` in total."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        model_path: Path,
        context_length: int,
        chat_format: str | None,
        library_version: str,
    ) -> str:
        """Identify a state; raises OSError if the model file can't be read."""
        stat = model_path.stat()
        identity = {
            "format": _FORMAT,
            "library": library_version,
            "model": model_path.name,
            "model_bytes": stat.st_size,
            "model_mtime_ns": stat.st_mtime_ns,
            "context_length": context_length,
            "chat_format": chat_format,
            "system_prompt": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]

    def entries(self) -> list[Path]:
        try:
            return sorted(self.directory.glob(f"*{SUFFIX}"))
        except OSError:
            return []

    def disk_bytes(self) -> int:
        total = 0
        for path in self.entries():
            with contextlib.suppress(OSError):
                total += path.stat().st_size
        return total

    def fits(self, size: int) -> bool:
        return size <= self.max_bytes

    def find(self, key: str) -> Path | None:
        """The entry for ``key`` if its checksum matches (a damaged one is deleted)."""
        for path in self.directory.glob(f"{key}.*{SUFFIX}"):
            expected = _crc_from_name(path)
            try:
                actual = file_crc32(path)
            except OSError:
                continue  # removed by another process meanwhile
            if expected is None or actual != expected:
                log.warning("Deleting damaged prompt cache file %s", path.name)
                self.discard(path)
                continue
            with contextlib.suppress(OSError):
                os.utime(path)  # most recently used: pruned last
            return path
        return None

    def store(self, key: str, write: Callable[[Path], bool]) -> Path | None:
        """Save an entry with ``write(temp_path)``, then enforce the size cap."""
        tmp = self.directory / f"{key}{_TMP_MARKER}{os.getpid()}"
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not write(tmp):
                log.warning("llama.cpp could not save the prompt state to %s", tmp)
                tmp.unlink(missing_ok=True)
                return None
            final = self.directory / f"{key}.{file_crc32(tmp):08x}{SUFFIX}"
            for old in self.directory.glob(f"{key}.*{SUFFIX}"):
                if old != final:
                    old.unlink(missing_ok=True)
            tmp.replace(final)
        except OSError as exc:
            log.warning("Could not save the prompt cache: %s", exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return None
        self.prune(keep=final)
        return final if final.exists() else None

    def discard(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete %s: %s", path, exc)

    def prune(self, keep: Path | None = None) -> None:
        """Delete stale temp files, then least recently used entries above the cap."""
        now = time.time()
        try:
            leftovers = list(self.directory.glob(f"*{_TMP_MARKER}*"))
        except OSError:
            leftovers = []
        for tmp in leftovers:
            try:
                if now - tmp.stat().st_mtime > _STALE_TMP_SECONDS:
                    tmp.unlink()
            except OSError:
                pass

        sized: list[tuple[float, int, Path]] = []
        for path in self.entries():
            try:
                stat = path.stat()
            except OSError:
                continue
            sized.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in sized)
        # Oldest first; the entry just written goes last so it survives if it fits.
        sized.sort(key=lambda item: (item[2] == keep, item[0]))
        for _, size, path in sized:
            if total <= self.max_bytes:
                break
            self.discard(path)
            total -= size


def _crc_from_name(path: Path) -> int | None:
    # "<key>.<crc32 hex>.llstate"
    parts = path.name.split(".")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1], 16)
    except ValueError:
        return None
