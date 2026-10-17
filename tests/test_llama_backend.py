"""Unit tests for the llama.cpp backend that need no model file."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jarvis.ai.llama_cpp_backend import AIUnavailableError, LlamaCppBackend, ModelLoadError
from jarvis.ai.model_files import ModelFileError


def test_missing_model_file(tmp_path: Path) -> None:
    backend = LlamaCppBackend(tmp_path / "missing.gguf")
    with pytest.raises(ModelFileError, match="not found"):
        backend.load()
    assert not backend.is_loaded


def test_non_gguf_file_is_rejected_before_llama_cpp(tmp_path: Path) -> None:
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"not a model")
    with pytest.raises(ModelFileError, match="not a GGUF"):
        LlamaCppBackend(path).load()


def test_missing_llama_cpp_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # makes `import llama_cpp` fail
    with pytest.raises(AIUnavailableError) as info:
        LlamaCppBackend(path).load()
    assert info.value.hint and "[llm]" in info.value.hint


def test_corrupt_gguf_reports_load_error(tmp_path: Path) -> None:
    pytest.importorskip("llama_cpp")
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    with pytest.raises(ModelLoadError, match="could not load"):
        LlamaCppBackend(path, gpu_layers=0).load()


def test_complete_requires_load(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="not loaded"):
        LlamaCppBackend(tmp_path / "m.gguf").complete_json([], {}, max_tokens=8, temperature=0.0)


def test_unload_when_not_loaded_is_a_no_op(tmp_path: Path) -> None:
    LlamaCppBackend(tmp_path / "m.gguf").unload()
