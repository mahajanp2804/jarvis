"""CLI handlers for ``jarvis memory ...``: inspect, export and erase the memory.

These work on the database file directly (also when ``[memory] enabled`` is
false, so a turned-off memory can still be inspected and erased) and are
typed by the user, so they don't go through the agent.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.memory.store import SECTIONS, MemoryStore, Section, local_time

EXIT_OK, EXIT_ERROR, EXIT_NOT_DONE = 0, 1, 3


def open_store(config: JarvisConfig) -> MemoryStore:
    memory = config.memory
    return MemoryStore(
        memory.resolved_path(), history_days=memory.history_days, max_history=memory.max_history
    )


def _print_error(exc: JarvisError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"  Hint: {exc.hint}", file=sys.stderr)


def show(config: JarvisConfig, *, limit: int = 10) -> int:
    store = open_store(config)
    try:
        state = "on" if config.memory.enabled else "off ([memory] enabled = false)"
        history = "on" if config.memory.record_history else "off"
        size = f"{store.path.stat().st_size / 1024:.0f} KB" if store.exists else "not created yet"
        counts = store.counts()
        print(f"Memory:       {state}; task history {history}")
        print(f"Database:     {store.path} ({size})")
        print(
            f"Keeps:        {config.memory.history_days} days of history, "
            f"at most {config.memory.max_history} tasks"
        )
        print("Never stored: approvals, passwords/keys/tokens, outside content")
        prefs = store.preferences()
        print(f"\nPreferences ({counts['preferences']}):")
        for pref in prefs:
            print(f"  {pref.name}: {pref.value}")
        if not prefs:
            print('  (none; say "remember that my editor is VS Code")')
        tasks = store.history(limit=limit)
        print(f"\nRecent tasks ({len(tasks)} of {counts['history']}):")
        for task in tasks:
            tools = ", ".join(s.tool for s in task.steps) or "no tools"
            print(f"  {local_time(task.started_at)}  {task.outcome:<8} {task.request}  [{tools}]")
        if not tasks:
            print("  (none)")
        return EXIT_OK
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    finally:
        store.close()


def export(config: JarvisConfig, path: Path | None) -> int:
    """Write everything stored as JSON to ``path`` (never overwritten) or stdout."""
    store = open_store(config)
    try:
        text = json.dumps(store.export(), indent=2, ensure_ascii=False) + "\n"
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    finally:
        store.close()
    if path is None:
        sys.stdout.write(text)
        return EXIT_OK
    path = path.expanduser()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(f"Error: {path} already exists; not overwriting.", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"Error: can't write {path}: {exc.strerror}", file=sys.stderr)
        return EXIT_ERROR
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Exported the memory to {path} (readable only by you; it contains your history).")
    return EXIT_OK


def clear(
    config: JarvisConfig,
    sections: Sequence[Section],
    *,
    assume_yes: bool,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    store = open_store(config)
    try:
        counts = store.counts()
        chosen = list(sections) or list(SECTIONS)
        if not any(counts[s] for s in chosen):
            print("Nothing to erase.")
            return EXIT_OK
        if not assume_yes:
            summary = " and ".join(_amount(s, counts[s]) for s in chosen)
            try:
                answer = (input_fn or input)(
                    f"Erase {summary} from {store.path}? This can't be undone. [y/N] "
                )
            except EOFError:
                answer = ""
            if answer.strip().lower() not in {"y", "yes"}:
                print("Nothing erased.")
                return EXIT_NOT_DONE
        removed = store.clear(chosen)
        print("Erased " + " and ".join(_amount(s, n) for s, n in removed.items()) + ".")
        return EXIT_OK
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    finally:
        store.close()


def _amount(section: str, n: int) -> str:
    one, many = {"preferences": ("preference", "preferences")}.get(
        section, ("history entry", "history entries")
    )
    return f"{n} {one if n == 1 else many}"


def remember(config: JarvisConfig, name: str, value: str) -> int:
    if not config.memory.enabled:
        print("Memory is turned off ([memory] enabled = false).", file=sys.stderr)
        return EXIT_NOT_DONE
    store = open_store(config)
    try:
        pref = store.remember(name, value)
        print(f"Remembered: {pref.name} = {pref.value}")
        return EXIT_OK
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    finally:
        store.close()


def forget(config: JarvisConfig, name: str) -> int:
    store = open_store(config)
    try:
        if store.forget(name):
            print(f"Forgot {name}.")
            return EXIT_OK
        print(f"Nothing is remembered as '{name}'.")
        return EXIT_NOT_DONE
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    finally:
        store.close()
