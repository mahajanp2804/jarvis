"""Command-line interface.

Examples::

    jarvis run "open Safari"
    jarvis --dry-run run "quit Music"
    jarvis shell
    jarvis apps list studio
    jarvis run "open my browser and the notes app"   # planned by the local model
    jarvis model download
    jarvis benchmark
    jarvis config init
    jarvis doctor
    jarvis service install          # start at login; shortcut ⌃⌥⌘J opens Jarvis
    jarvis ask "open my browser"    # send a request to the running service

Exit codes: 0 success (or dry run), 1 error, 3 not executed (denied,
cancelled, needs clarification, or not understood), 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

from jarvis import __version__, model_cli, service_cli
from jarvis.agent.orchestrator import Reply, format_result
from jarvis.ai.model_files import check_model_file
from jarvis.ai.registry import spec_for_filename
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime, build_runtime
from jarvis.config import JarvisConfig, load_config, resolve_config_path, write_example_config
from jarvis.errors import JarvisError
from jarvis.logging_setup import configure_logging
from jarvis.platforms import get_platform
from jarvis.platforms.base import DiagnosticCheck
from jarvis.runtime.permissions import InteractiveConfirmer

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_DONE = 3
EXIT_INTERRUPTED = 130

log = logging.getLogger("jarvis.cli")


def _common_options() -> argparse.ArgumentParser:
    # SUPPRESS defaults let these options appear before or after the subcommand
    # without the subparser overwriting a value given earlier.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=argparse.SUPPRESS, help="path to config.toml"
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="validate and describe actions without executing them",
    )
    common.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        default=argparse.SUPPRESS,
        help="override the configured log level",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Jarvis: a local personal computer assistant.",
        parents=[common],
    )
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = sub.add_parser("run", parents=[common], help='run one request, e.g. "open Safari"')
    run.add_argument("text", nargs="+", help="the request")

    sub.add_parser("shell", parents=[common], help="interactive prompt ('help', 'exit')")

    apps = sub.add_parser("apps", parents=[common], help="list, open or quit applications")
    apps_sub = apps.add_subparsers(dest="apps_command", required=True, metavar="ACTION")
    apps_list = apps_sub.add_parser("list", parents=[common], help="list installed applications")
    apps_list.add_argument("query", nargs="?", help="filter text")
    apps_list.add_argument("--limit", type=int, default=50)
    apps_open = apps_sub.add_parser("open", parents=[common], help="open an application")
    apps_open.add_argument("name", nargs="+")
    apps_quit = apps_sub.add_parser("quit", parents=[common], help="quit an application")
    apps_quit.add_argument("name", nargs="+")

    cfg = sub.add_parser("config", parents=[common], help="inspect or create configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True, metavar="ACTION")
    cfg_sub.add_parser("show", parents=[common], help="print the effective configuration")
    cfg_sub.add_parser("path", parents=[common], help="print the config file location")
    init = cfg_sub.add_parser("init", parents=[common], help="write an example config file")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")

    model = sub.add_parser("model", parents=[common], help="inspect, check or download the model")
    model_sub = model.add_subparsers(dest="model_command", required=True, metavar="ACTION")
    model_sub.add_parser("info", parents=[common], help="show model configuration and status")
    check = model_sub.add_parser("check", parents=[common], help="validate the model file")
    check.add_argument("--verify", action="store_true", help="also verify the SHA-256")
    download = model_sub.add_parser(
        "download", parents=[common], help="download the default model (asks first)"
    )
    download.add_argument("--yes", action="store_true", help="don't ask for confirmation")

    bench = sub.add_parser(
        "benchmark", parents=[common], help="measure model load time, inference and idle cost"
    )
    bench.add_argument("--repeats", type=int, default=2, help="rounds over the prompt set")
    bench.add_argument("--idle-seconds", type=float, default=10.0, help="idle sample length")
    bench.add_argument("--json", type=Path, help="also write results as JSON")

    daemon = sub.add_parser(
        "daemon", parents=[common], help="run the background service in the foreground"
    )
    daemon.add_argument(
        "--headless", action="store_true", help="no shortcut or dialogs (control socket only)"
    )
    service = sub.add_parser("service", parents=[common], help="manage the start-at-login service")
    service_sub = service.add_subparsers(dest="service_command", required=True, metavar="ACTION")
    for name, text in [
        ("install", "install the Launch Agent (start at login) and start it"),
        ("uninstall", "stop and remove the Launch Agent (keeps config, logs, models)"),
        ("start", "start the service"),
        ("stop", "stop the service gracefully"),
        ("restart", "stop and start the service"),
        ("status", "show service state, memory and CPU time"),
    ]:
        service_sub.add_parser(name, parents=[common], help=text)
    measure = service_sub.add_parser(
        "measure", parents=[common], help="sample the running service's idle cost"
    )
    measure.add_argument("--seconds", type=float, default=30.0)
    sub.add_parser("activate", parents=[common], help="open the request dialog of the service")
    ask = sub.add_parser("ask", parents=[common], help="send one request to the running service")
    ask.add_argument("text", nargs="+")

    sub.add_parser("doctor", parents=[common], help="check the environment and permissions")
    return parser


def _exit_code(results: Sequence[ToolResult], handled: bool = True) -> int:
    if not handled:
        return EXIT_NOT_DONE
    statuses = {r.status for r in results}
    if ToolStatus.FAILED in statuses:
        return EXIT_ERROR
    if statuses & {ToolStatus.DENIED, ToolStatus.CANCELLED, ToolStatus.NEEDS_CLARIFICATION}:
        return EXIT_NOT_DONE
    return EXIT_OK


def _print_error(exc: JarvisError) -> None:
    print(f"Error: {exc.message}", file=sys.stderr)
    if exc.hint:
        print(f"  Hint: {exc.hint}", file=sys.stderr)


def _apply_cli_overrides(config: JarvisConfig, args: argparse.Namespace) -> JarvisConfig:
    updates: dict[str, object] = {}
    if getattr(args, "dry_run", False):
        updates["dry_run"] = True
    if getattr(args, "log_level", None):
        updates["log_level"] = args.log_level
    if not updates:
        return config
    return config.model_copy(update={"general": config.general.model_copy(update=updates)})


def _reply_exit(reply: Reply) -> int:
    print(reply.text)
    return _exit_code(reply.results, reply.handled)


def _keep_background_processes(runtime: Runtime) -> None:
    """Background processes belong to this Jarvis process: keep it alive until Ctrl+C."""
    running = runtime.processes.running()
    if not running:
        return
    for proc in running:
        where = f" at {proc.url}" if proc.url else ""
        print(f"{proc.id}: `{proc.command}` is running{where}.")
    print("Press Ctrl+C to stop (or use `jarvis ask` with the background service).")
    try:
        while any(p.running for p in running):
            threading.Event().wait(1.0)
    except KeyboardInterrupt:
        print("\nStopping background processes…")


def _show_thinking() -> None:
    print("… thinking", file=sys.stderr, flush=True)


def _run_shell(runtime: Runtime, cancel: threading.Event) -> int:
    mode = " [dry run]" if runtime.config.general.dry_run else ""
    print(f"Jarvis {__version__}{mode}. Type 'help' for commands, 'exit' to leave.")
    while True:
        try:
            line = input("jarvis> ")
        except EOFError:
            print()
            return EXIT_OK
        except KeyboardInterrupt:
            print()
            continue
        if line.strip().lower() in {"exit", "quit", "bye", ":q"}:
            return EXIT_OK
        if line.strip().lower() in {"reset", ":reset"}:
            runtime.orchestrator.clear_history()
            print("Conversation history cleared.")
            continue
        try:
            print(runtime.orchestrator.handle(line).text)
        except KeyboardInterrupt:
            print("\nCancelled.")
            cancel.clear()  # the interrupt only cancels the current request


def _ai_checks(config: JarvisConfig) -> list[DiagnosticCheck]:
    if not config.ai.enabled:
        return [DiagnosticCheck("Local AI", None, "disabled ([ai] enabled = false)")]
    version = model_cli.llama_cpp_version()
    checks = [
        DiagnosticCheck(
            "llama-cpp-python",
            version is not None,
            version or 'not installed: CMAKE_ARGS="-DGGML_METAL=on" pip install -e ".[llm]"',
        )
    ]
    path = config.ai.resolved_model_path()
    try:
        report = check_model_file(path, spec_for_filename(path.name))
        checks.append(
            DiagnosticCheck("Model file", True, f"{path.name} ({report.size_bytes / 1e9:.2f} GB)")
        )
    except JarvisError as exc:
        checks.append(DiagnosticCheck("Model file", False, f"{exc.message} {exc.hint or ''}"))
    return checks


def _doctor(config: JarvisConfig, config_path: Path) -> int:
    checks = [
        DiagnosticCheck(
            "Python >= 3.11",
            sys.version_info >= (3, 11),
            f"{sys.version.split()[0]} at {sys.executable}",
        ),
        DiagnosticCheck(
            "Configuration",
            True,
            f"{config_path}" if config_path.is_file() else f"defaults (no file at {config_path})",
        ),
        DiagnosticCheck("Log directory", True, str(config.general.resolved_log_dir())),
        *_ai_checks(config),
    ]
    try:
        platform = get_platform()
    except JarvisError as exc:
        checks.append(DiagnosticCheck("Platform", False, exc.message))
    else:
        checks.append(DiagnosticCheck("Platform adapter", True, platform.name))
        checks.extend(platform.diagnostics())
        try:
            search = config.applications.search_paths or platform.apps.default_search_paths()
            count = len(platform.apps.discover(search))
            checks.append(DiagnosticCheck("Application discovery", count > 0, f"{count} apps"))
        except JarvisError as exc:
            checks.append(DiagnosticCheck("Application discovery", False, exc.message))

    symbols = {True: "✓", False: "✗", None: "•"}
    for check in checks:
        print(f" {symbols[check.ok]} {check.name}: {check.detail}")
    return EXIT_ERROR if any(c.ok is False for c in checks) else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    env = os.environ
    config_arg: Path | None = getattr(args, "config", None)

    if args.command == "config" and args.config_command in {"path", "init"}:
        path, _ = resolve_config_path(config_arg, env)
        if args.config_command == "path":
            print(path)
            return EXIT_OK
        try:
            write_example_config(path, overwrite=args.force)
        except JarvisError as exc:
            _print_error(exc)
            return EXIT_ERROR
        print(f"Wrote example configuration to {path}")
        return EXIT_OK

    try:
        config = _apply_cli_overrides(load_config(config_arg, env), args)
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR

    if args.command == "config":  # show
        print(f"# source: {resolve_config_path(config_arg, env)[0]}", file=sys.stderr)
        print(config.model_dump_json(indent=2))
        return EXIT_OK

    log_dir = config.general.resolved_log_dir()
    try:
        configure_logging(config.general.log_level, log_dir)
    except OSError as exc:
        print(f"Error: cannot write logs to {log_dir}: {exc.strerror}", file=sys.stderr)
        return EXIT_ERROR

    if args.command == "doctor":
        return _doctor(config, resolve_config_path(config_arg, env)[0])
    if args.command == "activate":
        return service_cli.activate()
    if args.command == "ask":
        return service_cli.ask(" ".join(args.text), config)
    if args.command == "model":
        if args.model_command == "info":
            return model_cli.model_info(config)
        if args.model_command == "check":
            return model_cli.model_check(config, verify=args.verify)
        return model_cli.model_download(config, assume_yes=args.yes)

    try:
        platform = get_platform()
    except JarvisError as exc:
        _print_error(exc)
        return EXIT_ERROR

    if args.command == "daemon":
        return service_cli.run_daemon(config, platform, headless=args.headless)
    if args.command == "service":
        return service_cli.service_command(
            args.service_command,
            config,
            platform,
            config_path=config_arg,
            measure_seconds=getattr(args, "seconds", 30.0),
        )
    if args.command == "benchmark":
        return model_cli.benchmark(
            config,
            platform,
            repeats=args.repeats,
            idle_seconds=args.idle_seconds,
            json_path=args.json,
        )

    cancel = threading.Event()
    runtime = build_runtime(config, platform, InteractiveConfirmer(), cancel_event=cancel)
    runtime.orchestrator.on_planning = _show_thinking
    try:
        if args.command == "run":
            code = _reply_exit(runtime.orchestrator.handle(" ".join(args.text)))
            _keep_background_processes(runtime)
            return code
        if args.command == "shell":
            return _run_shell(runtime, cancel)
        if args.command == "apps":
            if args.apps_command == "list":
                arguments: dict[str, object] = {"limit": args.limit}
                if args.query:
                    arguments["query"] = args.query
                call = ToolCall(tool="apps.list", arguments=arguments)
            else:
                call = ToolCall(
                    tool=f"apps.{args.apps_command}", arguments={"name": " ".join(args.name)}
                )
            result = runtime.dispatcher.execute(call)
            print(format_result(result))
            return _exit_code([result])
        raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover
    except KeyboardInterrupt:
        cancel.set()
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        runtime.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
