"""CLI handlers for the background service.

``jarvis daemon``            run the service in the foreground (what launchd runs)
``jarvis service install``   register the Launch Agent (start at login) and start it
``jarvis service start|stop|restart|status|uninstall``
``jarvis service measure``   sample the running service's idle CPU, wakeups, memory
``jarvis activate``          open the request dialog, as if the shortcut was pressed
``jarvis ask "..."``         send one request to the running service
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from jarvis.benchmark import combined_stats
from jarvis.config import JarvisConfig
from jarvis.errors import JarvisError
from jarvis.logging_setup import configure_logging
from jarvis.platforms.base import PlatformAdapter, ProcessStats
from jarvis.runtime.control import send_request
from jarvis.runtime.lifecycle import Daemon, DaemonPaths

EXIT_OK, EXIT_ERROR, EXIT_NOT_DONE = 0, 1, 3
_MB = 1024 * 1024

log = logging.getLogger(__name__)


def _print_error(exc: JarvisError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"  Hint: {exc.hint}", file=sys.stderr)


def run_daemon(config: JarvisConfig, platform: PlatformAdapter, *, headless: bool) -> int:
    configure_logging(
        config.general.log_level, config.general.resolved_log_dir(), console=sys.stderr.isatty()
    )
    try:
        return Daemon(config, platform, headless=headless).run()
    except JarvisError as exc:
        log.error("%s %s", exc.message, exc.hint or "")
        _print_error(exc)
        return EXIT_ERROR


def service_program(config_path: Path | None) -> list[str]:
    """Command launchd runs: this virtualenv's Python, so dependencies match."""
    program = [sys.executable, "-m", "jarvis", "daemon"]
    if config_path is not None:
        program += ["--config", str(config_path.expanduser().resolve())]
    return program


def service_command(
    action: str,
    config: JarvisConfig,
    platform: PlatformAdapter,
    *,
    config_path: Path | None,
    measure_seconds: float = 30.0,
) -> int:
    try:
        manager = platform.service_manager(config.service.label)
        if action == "install":
            log_file = config.general.resolved_log_dir() / "service.log"
            program = service_program(config_path)
            path = manager.install(
                program,
                {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONUNBUFFERED": "1"},
                log_file,
            )
            print(f"Installed {path}")
            print(f"  runs:  {' '.join(program)}")
            print(f"  log:   {log_file}")
            print("Jarvis now starts at login. macOS may show a 'Background Items Added' notice.")
            return _wait_and_report(config, platform)
        if action == "uninstall":
            manager.uninstall()
            print("Uninstalled. Config, logs and models are kept:")
            print(f"  {config.general.resolved_log_dir()}")
            print(f"  {config.ai.resolved_model_path().parent.parent}")
            return EXIT_OK
        if action == "start":
            manager.start()
            return _wait_and_report(config, platform)
        if action == "stop":
            was_running = manager.status().running
            manager.stop()
            print(
                "Stopped. Jarvis starts again at next login."
                if was_running
                else "Jarvis isn't running."
            )
            return EXIT_OK
        if action == "restart":
            manager.stop()
            manager.start()
            return _wait_and_report(config, platform)
        if action == "status":
            return show_status(config, platform)
        if action == "measure":
            return measure(platform, seconds=measure_seconds)
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    raise ValueError(f"unknown service action {action!r}")


def _query(timeout: float = 5.0) -> dict[str, Any] | None:
    try:
        return send_request(DaemonPaths.default().socket, {"cmd": "status"}, timeout=timeout)
    except JarvisError:
        return None


def _wait_and_report(config: JarvisConfig, platform: PlatformAdapter, timeout: float = 30.0) -> int:
    # launchd starts a job at most once per ThrottleInterval (10 s), so a quick
    # stop/start can take that long; say so instead of looking stuck.
    started = time.monotonic()
    noted = False
    while time.monotonic() - started < timeout:
        if _query(timeout=2) is not None:
            return show_status(config, platform)
        if not noted and time.monotonic() - started > 2:
            print("Waiting for launchd to start Jarvis (it allows one start per 10 s)…", flush=True)
            noted = True
        time.sleep(0.25)
    print(
        "The service did not answer yet. Check `jarvis service status` and "
        f"{config.general.resolved_log_dir() / 'service.log'}"
    )
    return EXIT_ERROR


def _combined_stats(platform: PlatformAdapter, pids: Sequence[int]) -> ProcessStats | None:
    return combined_stats(platform, pids)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m {secs}s"


def show_status(config: JarvisConfig, platform: PlatformAdapter) -> int:
    try:
        launchd = platform.service_manager(config.service.label).status()
    except JarvisError:
        launchd = None
    live = _query()
    if launchd is not None:
        state = (
            "running"
            if launchd.running
            else ("loaded, stopped" if launchd.loaded else "not loaded")
        )
        installed = "installed" if launchd.installed else "not installed"
        pid = f" (pid {launchd.pid})" if launchd.pid else ""
        print(f"Launch Agent:  {installed}, {state}{pid}")
        if launchd.last_exit and not launchd.running:
            print(f"Last exit:     {launchd.last_exit}")
    if live is None:
        print("Service:       not answering (not running)")
        return EXIT_NOT_DONE
    pids = [int(live["pid"]), *[int(p) for p in live.get("worker_pids", [])]]
    uptime = _duration(live["uptime_seconds"])
    print(f"Service:       Jarvis {live['version']}, {live['mode']}, up {uptime}")
    if live.get("mode") == "headless":
        shortcut = "none (headless mode)"
    else:
        shortcut = live.get("hotkey") or f"unavailable: {live.get('hotkey_error')}"
    print(f"Shortcut:      {shortcut}")
    voice = []
    if live.get("push_to_talk"):
        voice.append(f"push-to-talk {live['push_to_talk']}")
    if live.get("speak_replies"):
        voice.append("spoken replies")
    print(f"Voice:         {', '.join(voice) or 'off'}")
    model = "loaded" if live.get("model_loaded") else "not loaded (loads on demand)"
    if not live.get("ai_enabled"):
        model = "AI disabled"
    print(f"Model:         {model}; loads so far: {live.get('model_loads', 0)}")
    print(f"Requests:      {live.get('requests', 0)}{' (busy)' if live.get('busy') else ''}")
    stats = _combined_stats(platform, pids)
    if stats is not None:
        worker = f" incl. model worker {pids[1:]}" if len(pids) > 1 else ""
        print(
            f"Memory:        footprint {(stats.footprint_bytes or 0) / _MB:,.0f} MB, "
            f"RSS {stats.rss_bytes / _MB:,.0f} MB{worker}"
        )
        print(f"CPU time:      {stats.cpu_seconds or 0:.2f} s since start")
    return EXIT_OK


def measure(
    platform: PlatformAdapter,
    *,
    seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    query: Callable[[], dict[str, Any] | None] = _query,
) -> int:
    """Sample the running service from outside (it isn't disturbed)."""
    live = query()
    if live is None:
        print("Jarvis isn't running. Start it with `jarvis service start`.")
        return EXIT_NOT_DONE
    pids = [int(live["pid"]), *[int(p) for p in live.get("worker_pids", [])]]
    before = _combined_stats(platform, pids)
    if before is None:
        print("Process statistics aren't available on this platform.")
        return EXIT_ERROR
    print(f"Sampling pid(s) {pids} for {seconds:g}s…", file=sys.stderr)
    sleep(seconds)
    after = _combined_stats(platform, pids)
    if after is None:
        print("The service exited during the measurement.")
        return EXIT_ERROR
    cpu = ((after.cpu_seconds or 0) - (before.cpu_seconds or 0)) / seconds * 100
    wakeups = ((after.wakeups or 0) - (before.wakeups or 0)) / seconds
    model = "loaded" if live.get("model_loaded") else "not loaded"
    print(f"Model {model}; {seconds:g}s sample of pid(s) {pids}:")
    print(f"  CPU        {cpu:.3f}%")
    print(f"  Wakeups    {wakeups:.2f}/s")
    print(f"  Footprint  {(after.footprint_bytes or 0) / _MB:,.1f} MB")
    print(f"  RSS        {after.rss_bytes / _MB:,.1f} MB")
    return EXIT_OK


def activate() -> int:
    try:
        reply = send_request(DaemonPaths.default().socket, {"cmd": "activate"})
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    if not reply.get("ok"):
        print(f"Error: {reply.get('error')}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def ask(text: str, config: JarvisConfig) -> int:
    try:
        reply = send_request(
            DaemonPaths.default().socket,
            {"cmd": "ask", "text": text},
            timeout=config.service.request_timeout_seconds + 10,
        )
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR
    if not reply.get("ok"):
        print(f"Error: {reply.get('error')}", file=sys.stderr)
        return EXIT_ERROR
    print(reply.get("text", ""))
    statuses = set(reply.get("statuses", []))
    if not reply.get("handled", True):
        return EXIT_NOT_DONE
    if "failed" in statuses:
        return EXIT_ERROR
    if statuses & {"denied", "cancelled", "needs_clarification"}:
        return EXIT_NOT_DONE
    return EXIT_OK
