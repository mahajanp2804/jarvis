"""Terminal tools: run allowlisted commands in a project, manage dev servers.

* ``terminal.run``: one command in a project folder. Its risk comes from
  :class:`~jarvis.tools.commands.CommandPolicy`. Dev servers (and anything
  run with ``background: true``) start in the background; everything else
  runs to completion with a time limit, and the output plus a short diagnosis
  is returned (marked untrusted).
* ``terminal.list`` / ``terminal.output`` / ``terminal.stop``: manage
  background processes Jarvis started. They are stopped when Jarvis exits.
* ``terminal.open``: open Terminal.app in a folder, so you can take over.

Processes run in their own process group with a minimal environment (Jarvis's
own environment is not passed on), and honour Jarvis's cancel event.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import TerminalConfig
from jarvis.errors import ToolExecutionError
from jarvis.platforms.base import PlatformAdapter
from jarvis.tools.base import PreparedAction, ToolSpec
from jarvis.tools.commands import Classified, CommandPolicy, diagnose
from jarvis.tools.files import PathText
from jarvis.tools.paths import PathPolicy, display

log = logging.getLogger(__name__)

_URL = re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1?\])(?::\d+)?[^\s'\"<>]*")
_READY = re.compile(r"(?i)\b(ready|listening|running on|started server|compiled|serving)\b")


class RunArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=500, description="one command, no pipes")
    folder: PathText = Field(description="project folder to run it in")
    background: bool = Field(default=False, description="true for servers that keep running")


class ProcessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=20, description="process id from terminal.list")


class OutputArgs(ProcessArgs):
    lines: int = Field(default=40, ge=1, le=200)


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FolderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    folder: PathText


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "[… earlier output not shown]\n" + text[-limit:]


def _kill_group(proc: subprocess.Popen[bytes], grace: float = 3.0) -> None:
    """SIGTERM the process group, then SIGKILL after ``grace`` seconds."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=grace)
    except ProcessLookupError:
        pass


@dataclass
class ManagedProcess:
    id: str
    command: str
    folder: Path
    proc: subprocess.Popen[bytes]
    log_path: Path
    started: float = field(default_factory=time.time)
    url: str | None = None

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def output(self, max_chars: int = 4000) -> str:
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(max(0, self.log_path.stat().st_size - max_chars))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


class ProcessManager:
    """Background processes started by Jarvis; all stopped on :meth:`shutdown`."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._processes: dict[str, ManagedProcess] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def start(
        self, classified: Classified, command: str, folder: Path, env: dict[str, str]
    ) -> ManagedProcess:
        self._log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        pid_id = f"p{next(self._ids)}"
        log_path = self._log_dir / f"{pid_id}-{int(time.time())}.log"
        fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            proc = subprocess.Popen(
                classified.argv,
                cwd=folder,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=fd,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolExecutionError(f"Could not start `{command}`: {exc.strerror}") from exc
        finally:
            os.close(fd)
        managed = ManagedProcess(pid_id, command, folder, proc, log_path)
        with self._lock:
            self._processes[pid_id] = managed
        return managed

    def get(self, pid_id: str) -> ManagedProcess:
        with self._lock:
            managed = self._processes.get(pid_id)
        if managed is None:
            raise ToolExecutionError(
                f"No background process '{pid_id}'.", hint="See terminal.list for the ids."
            )
        return managed

    def all(self) -> list[ManagedProcess]:
        with self._lock:
            return list(self._processes.values())

    def running(self) -> list[ManagedProcess]:
        return [p for p in self.all() if p.running]

    def shutdown(self) -> None:
        for managed in self.running():
            log.info("Stopping background process %s (%s)", managed.id, managed.command)
            _kill_group(managed.proc)


class TerminalTools:
    """Builds the ``terminal.*`` tool specs."""

    def __init__(
        self,
        platform: PlatformAdapter,
        paths: PathPolicy,
        config: TerminalConfig,
        processes: ProcessManager,
        *,
        max_output_chars: int = 6000,
    ) -> None:
        self._platform = platform
        self._paths = paths
        self._config = config
        self._processes = processes
        self._commands = CommandPolicy(config.search_path, config.extra_programs)
        self._max_output = max_output_chars

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("terminal.run", "Run one development command in a project folder "
                     "(servers keep running in the background).", RunArgs, RiskLevel.MEDIUM,
                     self._prepare_run),
            ToolSpec("terminal.list", "List background processes Jarvis started.", EmptyArgs,
                     RiskLevel.READ_ONLY, self._prepare_list, idempotent=True),
            ToolSpec("terminal.output", "Show recent output of a background process.",
                     OutputArgs, RiskLevel.READ_ONLY, self._prepare_output, idempotent=True),
            ToolSpec("terminal.stop", "Stop a background process Jarvis started.", ProcessArgs,
                     RiskLevel.LOW, self._prepare_stop),
            ToolSpec("terminal.open", "Open the Terminal app in a folder.", FolderArgs,
                     RiskLevel.LOW, self._prepare_open),
        ]  # fmt: skip

    def environment(self, folder: Path) -> dict[str, str]:
        """Minimal environment: no secrets or settings leak from Jarvis's own."""
        env = {
            "PATH": os.pathsep.join(self._config.search_path),
            "HOME": str(self._paths.home),
            "USER": os.environ.get("USER", ""),
            "LANG": "en_US.UTF-8",
            "TERM": "dumb",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
        }
        venv = next(
            (folder / v for v in (".venv", "venv", "env") if (folder / v / "pyvenv.cfg").exists()),
            None,
        )
        if venv is not None:
            env["VIRTUAL_ENV"] = str(venv)
            env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env['PATH']}"
        return env

    # -- run ---------------------------------------------------------------------------------

    def _prepare_run(self, args: RunArgs) -> PreparedAction:
        folder = self._paths.readable(args.folder)
        if not folder.is_dir():
            raise ToolExecutionError(f"{display(folder, self._paths.home)} is not a folder.")
        classified = self._commands.classify(args.command, folder)
        background = args.background or classified.long_running
        risk = max(classified.risk, RiskLevel.MEDIUM) if background else classified.risk
        where = display(folder, self._paths.home)
        verb = "Start in the background" if background else "Run"
        warning = None
        if risk >= RiskLevel.MEDIUM:
            warning = f"This {classified.reason}."

        def run(cancel: threading.Event) -> ToolResult:
            env = self.environment(folder)
            if background:
                return self._start_background(classified, args.command, folder, env, cancel)
            return self._run_to_completion(classified, args.command, folder, env, cancel)

        return PreparedAction(
            description=f"{verb}: `{args.command}` in {where}",
            run=run,
            risk=risk,
            warning=warning,
        )

    def _run_to_completion(
        self,
        classified: Classified,
        command: str,
        folder: Path,
        env: dict[str, str],
        cancel: threading.Event,
    ) -> ToolResult:
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                classified.argv,
                cwd=folder,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolExecutionError(f"Could not run `{command}`: {exc.strerror}") from exc
        chunks: deque[bytes] = deque()
        size = [0]
        stream = proc.stdout
        if stream is None:  # pragma: no cover - stdout=PIPE guarantees a stream
            raise ToolExecutionError("No output stream.")

        def reader() -> None:
            while chunk := os.read(stream.fileno(), 65536):
                chunks.append(chunk)
                size[0] += len(chunk)
                while size[0] > self._max_output * 4 and len(chunks) > 1:
                    size[0] -= len(chunks.popleft())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        timeout = self._config.timeout_seconds
        stopped_reason = None
        while proc.poll() is None:
            if cancel.wait(0.2):
                stopped_reason = "Jarvis is stopping"
            elif time.monotonic() - started > timeout:
                stopped_reason = f"it was still running after {timeout:g}s"
            if stopped_reason:
                _kill_group(proc)
                break
        thread.join(timeout=2)
        output = b"".join(chunks).decode("utf-8", errors="replace")
        seconds = time.monotonic() - started
        code = proc.returncode
        data = {"exit_code": code, "seconds": round(seconds, 2), "command": command}
        shown = _tail(output.strip(), self._max_output) or "(no output)"
        if stopped_reason:
            timed_out = not cancel.is_set()
            return ToolResult(
                tool="terminal.run",
                status=ToolStatus.FAILED if timed_out else ToolStatus.CANCELLED,
                message=f"Stopped `{command}`: {stopped_reason}.\n{shown}",
                data=data,
                hint="If it's a server, run it with background: true." if timed_out else None,
                untrusted=True,
                verified=False,
            )
        if code == 0:
            return ToolResult(
                tool="terminal.run",
                status=ToolStatus.SUCCESS,
                message=f"`{command}` finished OK in {seconds:.1f}s.\n{shown}",
                data=data,
                untrusted=True,
                verified=True,
            )
        findings = diagnose(output)
        return ToolResult(
            tool="terminal.run",
            status=ToolStatus.FAILED,
            message=f"`{command}` failed (exit code {code}) after {seconds:.1f}s.\n{shown}",
            hint=" ".join(findings) or None,
            data={**data, "diagnosis": findings},
            untrusted=True,
            verified=False,
        )

    def _start_background(
        self,
        classified: Classified,
        command: str,
        folder: Path,
        env: dict[str, str],
        cancel: threading.Event,
    ) -> ToolResult:
        managed = self._processes.start(classified, command, folder, env)
        deadline = time.monotonic() + self._config.startup_seconds
        output = ""
        while time.monotonic() < deadline and managed.running:
            output = managed.output()
            match = _URL.search(output)
            if match:
                managed.url = match.group(0).rstrip(".,)")
                break
            if cancel.wait(0.25):
                break
        output = managed.output()
        data = {"id": managed.id, "pid": managed.proc.pid, "url": managed.url,
                "log": str(managed.log_path), "command": command}  # fmt: skip
        shown = _tail(output.strip(), 1500) or "(no output yet)"
        if not managed.running:
            exit_code = managed.proc.returncode
            findings = diagnose(output)
            return ToolResult(
                tool="terminal.run",
                status=ToolStatus.FAILED,
                message=f"`{command}` exited right away (exit code {exit_code}).\n{shown}",
                hint=" ".join(findings) or None,
                data={**data, "exit_code": exit_code, "diagnosis": findings},
                untrusted=True,
                verified=False,
            )
        ready = managed.url is not None or bool(_READY.search(output))
        where = f" at {managed.url}" if managed.url else ""
        status_text = "is running" if ready else "started (no ready message yet)"
        return ToolResult(
            tool="terminal.run",
            status=ToolStatus.SUCCESS,
            message=f"`{command}` {status_text}{where} as {managed.id}.\n{shown}",
            data=data,
            untrusted=True,
            verified=True if ready else None,
        )

    # -- manage ----------------------------------------------------------------------------

    def _prepare_list(self, args: EmptyArgs) -> PreparedAction:
        def run(cancel: threading.Event) -> ToolResult:
            items = []
            for p in self._processes.all():
                state = "running" if p.running else f"exited ({p.proc.returncode})"
                items.append({"id": p.id, "command": p.command, "state": state, "url": p.url,
                              "folder": display(p.folder, self._paths.home)})  # fmt: skip
            if not items:
                message = "No background processes."
            else:
                message = "Background processes:\n" + "\n".join(
                    f"  {i['id']}: `{i['command']}` in {i['folder']}: {i['state']}"
                    + (f" {i['url']}" if i["url"] else "")
                    for i in items
                )
            return ToolResult(tool="terminal.list", status=ToolStatus.SUCCESS, message=message,
                              data={"processes": items})  # fmt: skip

        return PreparedAction(description="List background processes", run=run)

    def _prepare_output(self, args: OutputArgs) -> PreparedAction:
        managed = self._processes.get(args.id)

        def run(cancel: threading.Event) -> ToolResult:
            lines = managed.output(200_000).splitlines()[-args.lines :]
            text = "\n".join(lines) or "(no output)"
            return ToolResult(tool="terminal.output", status=ToolStatus.SUCCESS,
                              message=f"Last output of {managed.id} (`{managed.command}`):\n{text}",
                              untrusted=True)  # fmt: skip

        return PreparedAction(description=f"Show output of {managed.id}", run=run)

    def _prepare_stop(self, args: ProcessArgs) -> PreparedAction:
        managed = self._processes.get(args.id)
        if not managed.running:

            def done(cancel: threading.Event) -> ToolResult:
                return ToolResult(
                    tool="terminal.stop",
                    status=ToolStatus.SUCCESS,
                    message=f"{managed.id} already stopped.",
                    verified=True,
                )

            return PreparedAction(description=f"{managed.id} already stopped", run=done, noop=True)

        def run(cancel: threading.Event) -> ToolResult:
            _kill_group(managed.proc)
            ok = not managed.running
            return ToolResult(
                tool="terminal.stop",
                status=ToolStatus.SUCCESS if ok else ToolStatus.FAILED,
                message=f"Stopped {managed.id} (`{managed.command}`)."
                if ok
                else f"{managed.id} is still running.",
                verified=ok,
            )

        return PreparedAction(description=f"Stop {managed.id} (`{managed.command}`)", run=run)

    def _prepare_open(self, args: FolderArgs) -> PreparedAction:
        folder = self._paths.readable(args.folder)
        if not folder.is_dir():
            raise ToolExecutionError(f"{display(folder, self._paths.home)} is not a folder.")
        shown = display(folder, self._paths.home)

        def run(cancel: threading.Event) -> ToolResult:
            self._platform.open_terminal(folder)
            return ToolResult(tool="terminal.open", status=ToolStatus.SUCCESS,
                              message=f"Opened Terminal in {shown}.")  # fmt: skip

        return PreparedAction(description=f"Open Terminal in {shown}", run=run)
