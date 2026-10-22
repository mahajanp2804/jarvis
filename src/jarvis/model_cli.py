"""CLI handlers for ``jarvis model ...`` and ``jarvis benchmark``.

Kept apart from :mod:`jarvis.main` so argument parsing and these longer
workflows can be read and tested separately.
"""

from __future__ import annotations

import importlib.metadata
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from jarvis.ai.model_files import (
    ModelFileError,
    check_model_file,
    download_model,
    missing_model_hint,
)
from jarvis.ai.prompt_cache import PromptCache
from jarvis.ai.registry import DEFAULT_MODEL, spec_for_filename
from jarvis.benchmark import combined_stats, render, run_benchmark
from jarvis.bootstrap import build_runtime
from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.platforms.base import PlatformAdapter
from jarvis.runtime.permissions import DenyAllConfirmer

EXIT_OK, EXIT_ERROR, EXIT_NOT_DONE = 0, 1, 3


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def llama_cpp_version() -> str | None:
    """Installed llama-cpp-python version, without importing it."""
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def model_info(config: JarvisConfig) -> int:
    ai = config.ai
    path = ai.resolved_model_path()
    spec = spec_for_filename(path.name)
    version = llama_cpp_version()
    rows = [
        ("AI enabled", str(ai.enabled)),
        ("Backend", ai.backend),
        ("llama-cpp-python", version or 'not installed (pip install -e ".[llm]")'),
        ("Model file", str(path)),
        ("Present", _gb(path.stat().st_size) if path.is_file() else "no"),
        ("Known model", f"{spec.display_name}, {spec.license}" if spec else "no (custom file)"),
        ("Context length", str(ai.context_length)),
        ("GPU layers", "all" if ai.gpu_layers < 0 else str(ai.gpu_layers)),
        ("Idle unload", f"{ai.idle_unload_seconds:g}s"),
        ("Prompt cache", _prompt_cache_summary(config)),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")
    return EXIT_OK


def _prompt_cache_summary(config: JarvisConfig) -> str:
    ai = config.ai
    if not ai.prompt_cache:
        return "off"
    cache = PromptCache(ai.resolved_prompt_cache_dir(), ai.prompt_cache_max_mb << 20)
    used = cache.disk_bytes() / (1 << 20)
    return f"{used:,.0f} of {ai.prompt_cache_max_mb:,} MB used, {cache.directory}"


def model_check(config: JarvisConfig, *, verify: bool) -> int:
    path = config.ai.resolved_model_path()
    spec = spec_for_filename(path.name)
    try:
        if verify and spec:
            print(f"Computing SHA-256 of {_gb(spec.size_bytes)}…", file=sys.stderr)
        report = check_model_file(path, spec, verify_hash=verify)
    except ModelFileError as exc:
        print(f"✗ {exc.message}")
        if exc.hint:
            print(f"  Hint: {exc.hint}")
        return EXIT_ERROR
    print(f"✓ {path} is a GGUF file ({_gb(report.size_bytes)}).")
    if spec is None:
        print("  Not a registry model, so its size and checksum can't be verified.")
    elif report.sha256_ok:
        print("  SHA-256 matches the pinned value.")
    else:
        print("  Size matches the pinned value (use --verify to check the SHA-256).")
    return EXIT_OK


class _ProgressPrinter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._last = 0.0
        self._first_bytes: int | None = None

    def __call__(self, done: int, total: int) -> None:
        now = self._clock()
        if self._first_bytes is None:
            self._first_bytes = done
        if now - self._last < 0.5 and done < total:
            return
        self._last = now
        rate = (done - self._first_bytes) / max(now - self._start, 1e-6) / 1e6
        print(
            f"\r  {done / total:6.1%}  {done / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:5.1f} MB/s",
            end="" if done < total else "\n",
            file=sys.stderr,
            flush=True,
        )


def model_download(
    config: JarvisConfig,
    *,
    assume_yes: bool,
    input_fn: Callable[[str], str] = input,
    downloader: Callable[..., Path] = download_model,
) -> int:
    spec = DEFAULT_MODEL
    dest = config.ai.resolved_model_path()
    if dest.name != spec.filename:
        print(
            f"[ai] model_path points to {dest}, which is not the registry model "
            f"({spec.filename}).\nDownload that file yourself, or unset model_path to use "
            "the default model."
        )
        return EXIT_ERROR

    if dest.is_file():
        try:
            check_model_file(dest, spec)
            print(f"{spec.display_name} is already at {dest}.")
            print("Run `jarvis model check --verify` to check its SHA-256.")
            return EXIT_OK
        except ModelFileError as exc:
            print(f"Existing file is unusable ({exc.message}); it will be replaced.")

    part = dest.with_name(dest.name + ".part")
    resume = f" (resuming from {_gb(part.stat().st_size)})" if part.exists() else ""
    print(
        f"Model:    {spec.display_name}\n"
        f"Licence:  {spec.license}  ({spec.page_url})\n"
        f"Size:     {_gb(spec.size_bytes)}{resume}\n"
        f"From:     {spec.url}\n"
        f"To:       {dest}",
        flush=True,
    )
    if not assume_yes:
        try:
            answer = input_fn("Download now? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Not downloaded.")
            return EXIT_NOT_DONE
    try:
        downloader(spec, dest, progress=_ProgressPrinter())
    except ModelFileError as exc:
        print(f"\nError: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"  Hint: {exc.hint}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Downloaded and verified (SHA-256): {dest}")
    return EXIT_OK


def benchmark(
    config: JarvisConfig,
    platform: PlatformAdapter,
    *,
    repeats: int,
    idle_seconds: float,
    json_path: Path | None,
) -> int:
    if not config.ai.enabled:
        print("AI is disabled ([ai] enabled = false); nothing to benchmark.")
        return EXIT_ERROR
    path = config.ai.resolved_model_path()
    if not path.is_file():
        print(f"Model file not found: {path}\n  Hint: {missing_model_hint(path)}")
        return EXIT_ERROR

    # Safety: the benchmark only plans, and its runtime is forced into dry-run with
    # all confirmations denied, so nothing on the system can change. The idle
    # timer is pushed out so it can't unload the model mid-measurement. The prompt
    # cache goes to an empty temporary directory: the run starts cold every time
    # and never evicts the user's saved prompts.
    with tempfile.TemporaryDirectory(prefix="jarvis-benchmark-") as tmp:
        cache_dir = Path(tmp) if config.ai.prompt_cache else None
        bench_config = config.model_copy(
            update={
                "general": config.general.model_copy(update={"dry_run": True}),
                "ai": config.ai.model_copy(
                    update={"idle_unload_seconds": 86400.0, "prompt_cache_dir": cache_dir}
                ),
            }
        )
        runtime = build_runtime(bench_config, platform, DenyAllConfirmer())
        if runtime.models is None or runtime.planner is None:  # pragma: no cover - enabled above
            return EXIT_ERROR
        runtime_models = runtime.models
        try:
            report = run_benchmark(
                runtime.models,
                runtime.planner,
                runtime.orchestrator,
                lambda: combined_stats(platform, [None, *runtime_models.backend.process_ids()]),
                repeats=repeats,
                idle_seconds=idle_seconds,
                progress=lambda msg: print(f"… {msg}", file=sys.stderr, flush=True),
                prompt_cache_dir=cache_dir,
            )
        except JarvisError as exc:
            print(f"Error: {exc.message}", file=sys.stderr)
            if exc.hint:
                print(f"  Hint: {exc.hint}", file=sys.stderr)
            return EXIT_ERROR
        finally:
            runtime.close()
    print(render(report))
    if json_path is not None:
        report.write_json(json_path)
        print(f"\nWrote {json_path}")
    return EXIT_OK
