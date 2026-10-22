"""On-disk prompt cache: file management, and the llama.cpp backend with a fake library."""

from __future__ import annotations

import ctypes
import os
import sys
import time
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from jarvis.ai.base import ChatMessage
from jarvis.ai.llama_cpp_backend import LlamaCppBackend
from jarvis.ai.prompt_cache import PromptCache, file_crc32, system_prefix_length
from jarvis.ai.subprocess_backend import SubprocessBackend
from jarvis.ai.worker import backend_from_settings
from jarvis.bootstrap import build_backend
from jarvis.config import JarvisConfig

MB = 1 << 20
SYSTEM = "You are a careful assistant. Answer with a JSON plan for the user's request. " * 2


def key(**overrides: Any) -> str:
    args: dict[str, Any] = {
        "system_prompt": SYSTEM,
        "model_path": overrides.pop("model_path"),
        "context_length": 4096,
        "chat_format": None,
        "library_version": "0.3.36",
    }
    return PromptCache.make_key(**{**args, **overrides})


def writer(data: bytes) -> Any:
    def write(path: Path) -> bool:
        path.write_bytes(data)
        return True

    return write


# -- keys ----------------------------------------------------------------------------


def test_key_changes_with_everything_that_affects_the_state(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 64)
    base = key(model_path=model)
    assert key(model_path=model) == base
    variants = {
        key(model_path=model, system_prompt=SYSTEM + "Installed applications: Xcode."),
        key(model_path=model, context_length=8192),
        key(model_path=model, chat_format="chatml"),
        key(model_path=model, library_version="0.3.37"),
    }
    model.write_bytes(b"GGUF" + b"\0" * 65)  # replaced model file
    variants.add(key(model_path=model))
    assert base not in variants and len(variants) == 5


def test_key_needs_the_model_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        key(model_path=tmp_path / "missing.gguf")


# -- files ---------------------------------------------------------------------------


def test_store_then_find(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path / "cache", 10 * MB)
    saved = cache.store("k1", writer(b"state" * 100))
    assert saved is not None and saved.name == f"k1.{zlib.crc32(b'state' * 100):08x}.llstate"
    assert cache.find("k1") == saved
    assert cache.find("other") is None
    assert cache.entries() == [saved] and cache.disk_bytes() == 500
    assert not list((tmp_path / "cache").glob("*.tmp-*"))
    assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700


def test_find_deletes_a_damaged_file(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path, 10 * MB)
    saved = cache.store("k1", writer(b"state" * 100))
    assert saved is not None
    data = bytearray(saved.read_bytes())
    data[10] ^= 0xFF  # a flipped byte that llama.cpp itself would not notice
    saved.write_bytes(bytes(data))
    assert cache.find("k1") is None
    assert not saved.exists()


def test_find_ignores_files_without_a_checksum(tmp_path: Path) -> None:
    (tmp_path / "k1.llstate").write_bytes(b"x")
    assert PromptCache(tmp_path, MB).find("k1") is None


def test_storing_again_replaces_the_entry(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path, 10 * MB)
    first = cache.store("k1", writer(b"one"))
    second = cache.store("k1", writer(b"two"))
    assert first is not None and second is not None and first != second
    assert cache.entries() == [second]


def test_failed_write_leaves_nothing(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path, 10 * MB)

    def fail(path: Path) -> bool:
        path.write_bytes(b"partial")
        return False

    assert cache.store("k1", fail) is None
    assert list(tmp_path.iterdir()) == []


def test_unwritable_directory_is_not_fatal(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert PromptCache(blocker / "cache", MB).store("k1", writer(b"x")) is None


def test_least_recently_used_entries_are_pruned(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path, 250)
    old = cache.store("old", writer(b"a" * 100))
    used = cache.store("used", writer(b"b" * 100))
    assert old is not None and used is not None
    past = time.time() - 100
    os.utime(old, (past, past))
    os.utime(used, (past - 50, past - 50))
    assert cache.find("used") == used  # reading refreshes it
    new = cache.store("new", writer(b"c" * 100))
    assert new is not None
    assert cache.entries() == sorted([new, used])
    assert cache.disk_bytes() <= 250


def test_entry_larger_than_the_cap_is_not_kept(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path, 10)
    assert not cache.fits(11)
    assert cache.store("big", writer(b"x" * 11)) is None
    assert cache.entries() == []


def test_stale_temp_files_are_cleaned_up(tmp_path: Path) -> None:
    stale = tmp_path / "k1.tmp-123"
    fresh = tmp_path / "k2.tmp-456"  # another process may be writing it right now
    stale.write_bytes(b"x")
    fresh.write_bytes(b"y")
    past = time.time() - 3600
    os.utime(stale, (past, past))
    PromptCache(tmp_path, MB).prune()
    assert not stale.exists() and fresh.exists()


# -- prefix length -------------------------------------------------------------------


def detok(tokens: Sequence[int]) -> bytes:
    return bytes(tokens)


def test_prefix_ends_inside_the_system_prompt() -> None:
    prompt = f"<sys>{SYSTEM}</sys><user>open notes</user><bot>".encode()
    n = system_prefix_length(list(prompt), SYSTEM, detok)
    assert prompt[:n] == f"<sys>{SYSTEM}".encode()


def test_prefix_without_the_system_prompt_is_zero() -> None:
    assert system_prefix_length(list(b"<user>hi</user>"), SYSTEM, detok) == 0
    assert system_prefix_length(list(b"<sys></sys>"), "", detok) == 0


def test_prefix_with_multi_byte_tokens() -> None:
    pieces = [b"<sys>", b"You are", b" a careful", b" helper.", b"</sys>", b"<user>", b"hi"]
    table = dict(enumerate(pieces))

    def detok_pieces(tokens: Sequence[int]) -> bytes:
        return b"".join(table[t] for t in tokens)

    assert system_prefix_length(list(table), "You are a careful helper.", detok_pieces) == 4


# -- backend with a fake llama.cpp ---------------------------------------------------


class FakeLlama:
    """Byte-per-token stand-in for ``llama_cpp.Llama``."""

    instances: ClassVar[list[FakeLlama]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.n_ctx = kwargs["n_ctx"]
        self.input_ids = [0] * self.n_ctx
        self.n_tokens = 0
        self._ctx = SimpleNamespace(ctx=self)
        self.evaluated: list[int] = []  # prompt tokens read per request
        self.resets = 0
        FakeLlama.instances.append(self)

    def create_chat_completion(self, messages: list[ChatMessage], **_: Any) -> dict[str, Any]:
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<bot>"
        prompt = list(text.encode())
        reply = '{"ok": true}'
        reused = 0
        while reused < min(self.n_tokens, len(prompt)) and self.input_ids[reused] == prompt[reused]:
            reused += 1
        self.evaluated.append(len(prompt) - reused)
        tokens = prompt + list(reply.encode())
        self.input_ids[: len(tokens)] = tokens
        self.n_tokens = len(tokens)
        return {
            "choices": [{"message": {"content": reply}}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(reply)},
        }

    def detokenize(self, tokens: list[int], special: bool = False) -> bytes:
        return bytes(tokens)

    def reset(self) -> None:
        self.n_tokens = 0
        self.resets += 1

    def close(self) -> None:
        pass


class FakeLlamaCpp:
    """The parts of the ``llama_cpp`` module the backend calls."""

    __version__ = "0.0-test"
    llama_token = ctypes.c_int32
    LlamaGrammar = SimpleNamespace(from_json_schema=lambda schema, verbose=False: object())

    def __init__(self) -> None:
        self.Llama = FakeLlama
        self.state_size = 1000
        self.load_ok = True
        self.loads: list[str] = []
        self.saves: list[tuple[str, list[int]]] = []

    def llama_perf_context_reset(self, ctx: FakeLlama) -> None:
        pass

    def llama_perf_context(self, ctx: FakeLlama) -> Any:
        return SimpleNamespace(t_p_eval_ms=1.0, t_eval_ms=1.0, n_p_eval=ctx.evaluated[-1])

    def llama_get_memory(self, ctx: FakeLlama) -> FakeLlama:
        return ctx

    def llama_memory_seq_rm(self, mem: FakeLlama, seq: int, p0: int, p1: int) -> bool:
        assert (seq, p1) == (-1, -1)
        mem.n_tokens = min(mem.n_tokens, p0)
        return True

    def llama_state_get_size(self, ctx: FakeLlama) -> int:
        return self.state_size

    def llama_state_save_file(self, ctx: FakeLlama, path: bytes, tokens: Any, n: int) -> bool:
        assert ctx.n_tokens == n  # only the system prompt is in the KV cache
        self.saves.append((os.fsdecode(path), list(tokens[:n])))
        Path(os.fsdecode(path)).write_bytes(bytes(tokens[:n]))
        return True

    def llama_state_load_file(
        self, ctx: FakeLlama, path: bytes, tokens: Any, capacity: int, count: Any
    ) -> bool:
        self.loads.append(os.fsdecode(path))
        if not self.load_ok:
            return False
        data = Path(os.fsdecode(path)).read_bytes()
        tokens[: len(data)] = list(data)
        count._obj.value = len(data)
        return True


@pytest.fixture
def fake_lib(monkeypatch: pytest.MonkeyPatch) -> FakeLlamaCpp:
    lib = FakeLlamaCpp()
    monkeypatch.setitem(sys.modules, "llama_cpp", lib)
    FakeLlama.instances.clear()
    return lib


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


def ask(backend: LlamaCppBackend, request: str, system: str = SYSTEM) -> Any:
    messages: list[ChatMessage] = [
        {"role": "system", "content": system},
        {"role": "user", "content": request},
    ]
    return backend.complete_json(messages, {}, max_tokens=8, temperature=0.0)


def test_first_request_saves_and_a_reload_restores(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    cache = PromptCache(tmp_path / "cache", MB)
    backend = LlamaCppBackend(model, context_length=4096, prompt_cache=cache)
    backend.load()
    first = ask(backend, "open notes")
    assert first.prompt_cache == "saved" and first.prompt_cache_seconds is not None
    assert first.cached_prompt_tokens == 0
    [(_, saved_tokens)] = fake_lib.saves
    assert bytes(saved_tokens) == f"<system>{SYSTEM}".encode()  # no request on disk
    assert len(cache.entries()) == 1

    second = ask(backend, "quit music")  # same prompt, same process: memory only
    assert second.prompt_cache is None
    assert second.cached_prompt_tokens == len(saved_tokens)
    assert len(fake_lib.saves) == 1 and fake_lib.loads == []

    backend.unload()
    backend.load()
    third = ask(backend, "open safari")
    assert third.prompt_cache == "restored"
    assert third.cached_prompt_tokens == len(saved_tokens)
    assert len(fake_lib.loads) == 1 and len(fake_lib.saves) == 1


def test_changed_system_prompt_gets_its_own_entry(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    cache = PromptCache(tmp_path, MB)
    backend = LlamaCppBackend(model, prompt_cache=cache)
    backend.load()
    assert ask(backend, "open notes").prompt_cache == "saved"
    assert ask(backend, "open notes", SYSTEM + " Installed: Xcode.").prompt_cache == "saved"
    assert len(cache.entries()) == 2
    assert ask(backend, "open notes").prompt_cache == "restored"


def test_unreadable_state_is_discarded_and_rebuilt(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    cache = PromptCache(tmp_path, MB)
    backend = LlamaCppBackend(model, prompt_cache=cache)
    backend.load()
    ask(backend, "open notes")
    backend.unload()

    fake_lib.load_ok = False  # e.g. written by an incompatible llama.cpp
    backend.load()
    completion = ask(backend, "open notes")
    assert len(fake_lib.loads) == 1
    assert FakeLlama.instances[-1].resets == 1
    assert completion.prompt_cache == "saved" and completion.cached_prompt_tokens == 0
    [entry] = cache.entries()
    assert entry.name.split(".")[1] == f"{file_crc32(entry):08x}"


def test_state_larger_than_the_cap_is_not_saved(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    fake_lib.state_size = 2 * MB
    backend = LlamaCppBackend(model, prompt_cache=PromptCache(tmp_path, MB))
    backend.load()
    assert ask(backend, "open notes").prompt_cache is None
    assert ask(backend, "quit music").prompt_cache is None  # no retry for the same prompt
    assert fake_lib.saves == [] and list(tmp_path.glob("*.llstate")) == []


def test_requests_without_a_system_prompt_are_not_cached(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    backend = LlamaCppBackend(model, prompt_cache=PromptCache(tmp_path, MB))
    backend.load()
    completion = backend.complete_json(
        [{"role": "user", "content": "hello " * 50}], {}, max_tokens=8, temperature=0.0
    )
    assert completion.prompt_cache is None and fake_lib.saves == []


def test_short_system_prompts_are_not_cached(
    fake_lib: FakeLlamaCpp, model: Path, tmp_path: Path
) -> None:
    backend = LlamaCppBackend(model, prompt_cache=PromptCache(tmp_path, MB))
    backend.load()
    assert ask(backend, "open notes", system="Be brief.").prompt_cache is None
    assert fake_lib.saves == []


def test_cache_disabled(fake_lib: FakeLlamaCpp, model: Path) -> None:
    backend = LlamaCppBackend(model)
    backend.load()
    assert ask(backend, "open notes").prompt_cache is None
    assert fake_lib.saves == [] and fake_lib.loads == []


# -- configuration -------------------------------------------------------------------


def test_config_reaches_both_backends(tmp_path: Path) -> None:
    ai = JarvisConfig.model_validate(
        {"ai": {"prompt_cache_dir": str(tmp_path), "prompt_cache_max_mb": 300}}
    ).ai
    worker = build_backend(ai)
    assert isinstance(worker, SubprocessBackend)
    assert worker.settings["prompt_cache_dir"] == str(tmp_path)
    in_worker = backend_from_settings(worker.settings)
    assert in_worker.prompt_cache is not None
    assert in_worker.prompt_cache.directory == tmp_path
    assert in_worker.prompt_cache.max_bytes == 300 * MB

    in_process = build_backend(ai.model_copy(update={"isolation": "in_process"}))
    assert isinstance(in_process, LlamaCppBackend) and in_process.prompt_cache is not None

    off = build_backend(ai.model_copy(update={"prompt_cache": False}))
    assert isinstance(off, SubprocessBackend)
    assert backend_from_settings(off.settings).prompt_cache is None


def test_default_cache_location_is_outside_the_repo() -> None:
    path = JarvisConfig().ai.resolved_prompt_cache_dir()
    if sys.platform == "darwin":
        assert path == Path.home() / "Library/Application Support/Jarvis/prompt-cache"
    assert Path(__file__).resolve().parents[1] not in path.parents
