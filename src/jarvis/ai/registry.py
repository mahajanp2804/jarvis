"""Known, tested model files.

Each entry pins a repository *revision* together with the file's size and
SHA-256, so ``jarvis model download`` always fetches exactly the bytes Jarvis
was tested with. Other GGUF models can still be used via ``ai.model_path``.
They are just not downloadable or verifiable through Jarvis.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A downloadable model file pinned to an exact revision and checksum."""

    key: str
    display_name: str
    repo: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def page_url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


DEFAULT_MODEL = ModelSpec(
    key="qwen3-4b-instruct-2507-q4_k_m",
    display_name="Qwen3-4B-Instruct-2507 (Q4_K_M)",
    repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    revision="a06e946bb6b655725eafa393f4a9745d460374c9",
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    size_bytes=2_497_281_120,
    sha256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
    license="Apache-2.0",
)

KNOWN_MODELS: dict[str, ModelSpec] = {DEFAULT_MODEL.key: DEFAULT_MODEL}


def spec_for_filename(filename: str) -> ModelSpec | None:
    """Registry entry whose file name matches, if any."""
    return next((m for m in KNOWN_MODELS.values() if m.filename == filename), None)
