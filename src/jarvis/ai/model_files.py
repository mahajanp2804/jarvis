"""Model file checks and the explicit, user-initiated download.

Jarvis never downloads a model by itself. :func:`download_model` runs only when
the user runs ``jarvis model download`` and confirms. Downloads go to a
``.part`` file, resume after interruptions (HTTP Range), are verified against
the pinned SHA-256, and are only then renamed into place.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from jarvis import __version__
from jarvis.ai.registry import ModelSpec
from jarvis.errors import JarvisError

log = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
_CHUNK = 1024 * 1024


class ModelFileError(JarvisError):
    """The model file is missing, unreadable, or not what it should be."""


@dataclass(frozen=True)
class ModelFileReport:
    path: Path
    size_bytes: int
    expected_size: int | None
    sha256_ok: bool | None  # None = not checked


def missing_model_hint(path: Path) -> str:
    return (
        f"Download the default model with `jarvis model download`, or set "
        f"[ai] model_path in config.toml to a GGUF file. Expected: {path}"
    )


def check_model_file(
    path: Path, spec: ModelSpec | None = None, *, verify_hash: bool = False
) -> ModelFileReport:
    """Validate that ``path`` looks like a usable GGUF model.

    Checks existence, the GGUF magic number and, for registry models, the exact
    size. ``verify_hash`` also computes the SHA-256 (about 2 s per GB on an M4).
    """
    if not path.is_file():
        raise ModelFileError(f"Model file not found: {path}", hint=missing_model_hint(path))
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        raise ModelFileError(f"Cannot read model file {path}: {exc.strerror}") from exc
    if magic != GGUF_MAGIC:
        raise ModelFileError(
            f"{path} is not a GGUF model file.",
            hint="llama.cpp needs .gguf files; convert or download a GGUF build of the model.",
        )

    size = path.stat().st_size
    if spec is not None and size != spec.size_bytes:
        raise ModelFileError(
            f"{path.name} is {size:,} bytes but should be {spec.size_bytes:,}.",
            hint="The file is incomplete or a different build. Re-run `jarvis model download`.",
        )

    sha_ok: bool | None = None
    if verify_hash and spec is not None:
        sha_ok = sha256_file(path) == spec.sha256
        if not sha_ok:
            raise ModelFileError(
                f"SHA-256 of {path.name} does not match the pinned value.",
                hint="Delete the file and re-run `jarvis model download`.",
            )
    return ModelFileReport(path, size, spec.size_bytes if spec else None, sha_ok)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


Opener = Callable[..., Any]
Progress = Callable[[int, int], None]


def download_model(
    spec: ModelSpec,
    dest: Path,
    *,
    opener: Opener = urllib.request.urlopen,
    progress: Progress | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download ``spec`` to ``dest`` (resuming a previous partial download).

    Raises:
        ModelFileError: on network errors, unexpected responses or checksum mismatch.
    """
    if not spec.url.startswith("https://"):
        raise ModelFileError(f"Refusing non-HTTPS model URL: {spec.url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > spec.size_bytes:
        part.unlink()
        offset = 0

    if offset < spec.size_bytes:
        headers = {"User-Agent": f"jarvis-assistant/{__version__}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(spec.url, headers=headers)  # noqa: S310 - https enforced above
        try:
            with opener(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if offset and status != 206:
                    # Server ignored the Range header; start again from zero.
                    log.info("Server did not resume (HTTP %s); restarting download", status)
                    offset = 0
                mode = "ab" if offset else "wb"
                if progress:
                    progress(offset, spec.size_bytes)  # baseline for rate reporting
                with part.open(mode) as out:
                    _copy(response, out, offset, spec.size_bytes, progress)
        except urllib.error.HTTPError as exc:
            raise ModelFileError(f"Download failed: HTTP {exc.code} from {spec.url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ModelFileError(
                f"Download interrupted: {reason}",
                hint="Run `jarvis model download` again to resume.",
            ) from exc

    size = part.stat().st_size
    if size != spec.size_bytes:
        raise ModelFileError(
            f"Downloaded {size:,} of {spec.size_bytes:,} bytes.",
            hint="Run `jarvis model download` again to resume.",
        )
    if sha256_file(part) != spec.sha256:
        part.unlink()
        raise ModelFileError(
            "Downloaded file failed SHA-256 verification and was deleted.",
            hint="Run `jarvis model download` again.",
        )
    part.replace(dest)
    return dest


def _copy(
    response: IO[bytes], out: IO[Any], offset: int, total: int, progress: Progress | None
) -> None:
    written = offset
    while chunk := response.read(_CHUNK):
        out.write(chunk)
        written += len(chunk)
        if progress:
            progress(written, total)
