"""The on-device speech-to-text model (whisper.cpp), pinned like the language model.

Default: Whisper base.en, 5-bit quantised (60 MB). Measured on an M4 with
synthesized commands: every test sentence transcribed exactly, in ~0.2 s per
sentence on the CPU (0.08 s with ``[voice] stt_use_gpu``, whose first use after
a macOS shader-cache reset compiles Metal shaders for ~7 s); loading takes
~0.07 s. Other whisper.cpp ``ggml-*.bin`` models work via
``[voice] stt_model_path`` but can't be verified by Jarvis.
"""

from __future__ import annotations

from pathlib import Path

from jarvis.ai.model_files import ModelFileError
from jarvis.ai.registry import ModelSpec

DEFAULT_STT_MODEL = ModelSpec(
    key="whisper-base.en-q5_1",
    display_name="Whisper base.en (whisper.cpp, q5_1)",
    repo="ggerganov/whisper.cpp",
    revision="5359861c739e955e79d9a303bcbc70fb988958b1",
    filename="ggml-base.en-q5_1.bin",
    size_bytes=59_721_011,
    sha256="4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f",
    license="MIT",
)

GGML_MAGIC = b"lmgg"  # 0x67676d6c little-endian


def check_stt_model(path: Path, spec: ModelSpec | None = None) -> None:
    """Existence, the ggml magic number and (for the pinned model) the exact size."""
    if not path.is_file():
        raise ModelFileError(
            f"Speech model not found: {path}",
            hint="Download it with `jarvis voice download` (60 MB), or set "
            "[voice] stt_model_path to a whisper.cpp ggml-*.bin file.",
        )
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        raise ModelFileError(f"Cannot read {path}: {exc.strerror}") from exc
    if magic != GGML_MAGIC:
        raise ModelFileError(f"{path} is not a whisper.cpp (ggml) model file.")
    if spec is not None and path.stat().st_size != spec.size_bytes:
        raise ModelFileError(
            f"{path.name} is incomplete or a different build.",
            hint="Delete it and run `jarvis voice download` again.",
        )
