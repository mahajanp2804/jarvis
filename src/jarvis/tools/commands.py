"""Command policy for the terminal tools: parse, classify, deny, diagnose.

There is no shell. A command string is split into an argument list with
``shlex`` and executed directly, so ``|``, ``&&``, ``>``, ``$(...)`` and the like
are refused, not interpreted. The program must be on the allowlist below
(or in ``[terminal] extra_programs``). Each rule returns the *risk of that
specific call*:

* READ_ONLY: inspection (``git status``, ``pip list``, ``--version``)
* LOW: running tests (``pytest``, ``npm test``); no confirmation by default
* MEDIUM: running project code or changing the repo (``python app.py``,
  ``npm run dev``, ``git commit``); asks first
* HIGH: installs, history rewrites, remote changes (``pip install``,
  ``git push``, ``git reset --hard``); always asks
* refused: privilege, deletion, shells, inline code, network fetchers,
  system configuration (``sudo``, ``rm``, ``bash -c``, ``python -c``,
  ``curl``, ``launchctl``, ``git -c``)
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from jarvis.api.schemas import RiskLevel
from jarvis.errors import PolicyDeniedError, ToolInputError

R = RiskLevel

SHELL_TOKENS = frozenset({"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "&>"})
_SHELL_CHARS = re.compile(r"[`]|\$\(|\$\{")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_DENIED_GROUPS: tuple[tuple[str, str], ...] = (
    ("runs commands with administrator rights", "sudo su doas"),
    ("deletes files (use files.trash, which is recoverable)", "rm rmdir shred srm"),
    (
        "is a shell that can run anything",
        "sh bash zsh fish dash ksh csh tcsh env xargs eval exec nohup time watch",
    ),
    (
        "changes system settings or services",
        "launchctl defaults systemsetup networksetup pmset scutil csrutil spctl tccutil "
        "diskutil dd mkfs fdisk mount umount shutdown reboot halt",
    ),
    ("changes file permissions or ownership", "chmod chown chgrp chflags xattr"),
    (
        "downloads or uploads data (use the browser tools)",
        "curl wget nc ncat telnet ftp ssh scp sftp rsync",
    ),
    (
        "controls other apps or processes (use the app and terminal tools)",
        "osascript open kill killall pkill",
    ),
    ("runs inline code", "perl ruby php lua awk gawk"),
)
DENIED_PROGRAMS: dict[str, str] = {
    program: reason for reason, programs in _DENIED_GROUPS for program in programs.split()
}

# Programs that run project code when given a file or module.
INTERPRETERS = frozenset({"python", "python3", "node", "deno", "bun"})


@dataclass(frozen=True)
class Classified:
    argv: list[str]  # argv[0] is an absolute executable path
    program: str  # basename, e.g. "python3"
    risk: RiskLevel
    reason: str
    long_running: bool = False


class CommandPolicy:
    """Validates and classifies one command for a given working directory."""

    def __init__(self, search_path: Sequence[str], extra_programs: Sequence[str] = ()) -> None:
        self.search_path = os.pathsep.join(search_path)
        self.extra_programs = frozenset(extra_programs)

    def classify(self, command: str, cwd: Path) -> Classified:
        argv = self._split(command)
        executable, program = self._resolve(argv[0], cwd)
        args = argv[1:]
        if program in DENIED_PROGRAMS:
            raise PolicyDeniedError(
                f"Not allowed: `{program}` {DENIED_PROGRAMS[program]}.",
                hint="Jarvis only runs development commands from its allowlist.",
            )
        rule = RULES.get(_family(program))
        if rule is not None:
            risk, reason = rule(program, args)
        elif program in self.extra_programs:
            risk, reason = R.MEDIUM, "program allowed in [terminal] extra_programs"
        elif "/" in argv[0]:
            risk, reason = R.MEDIUM, "runs a script from this project"
        else:
            raise PolicyDeniedError(
                f"Not allowed: `{program}` is not on the terminal allowlist.",
                hint="Add it to [terminal] extra_programs in config.toml if you trust it.",
            )
        return Classified(
            argv=[executable, *args],
            program=program,
            risk=risk,
            reason=reason,
            long_running=looks_long_running(program, args),
        )

    @staticmethod
    def _split(command: str) -> list[str]:
        if not command.strip():
            raise ToolInputError("The command is empty.")
        if "\n" in command or "\x00" in command:
            raise ToolInputError("Give one command on one line.")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolInputError(f"Can't parse the command: {exc}.") from exc
        for token in argv:
            if token in SHELL_TOKENS or _SHELL_CHARS.search(token):
                raise PolicyDeniedError(
                    f"Not allowed: `{token}` is shell syntax; Jarvis runs one plain command.",
                    hint="Run each command separately; pipes and redirection aren't supported.",
                )
        if _ENV_ASSIGNMENT.match(argv[0]):
            raise PolicyDeniedError("Not allowed: setting environment variables in the command.")
        return argv

    def _resolve(self, first: str, cwd: Path) -> tuple[str, str]:
        """Absolute executable path and program name. Project paths must stay in cwd."""
        if "/" in first:
            candidate = Path(os.path.normpath(cwd / first))  # don't follow symlinks here
            if not (candidate == cwd or candidate.is_relative_to(cwd)):
                raise PolicyDeniedError(
                    f"Not allowed: `{first}` is outside the project folder.",
                    hint="Run programs by name, or from the project (e.g. .venv/bin/python).",
                )
            if not (candidate.is_file() and os.access(candidate, os.X_OK)):
                raise ToolInputError(f"`{first}` doesn't exist or isn't executable.")
            return str(candidate), _program_name(candidate.name)
        program = _program_name(first)
        if program in DENIED_PROGRAMS:
            return first, program  # refused by classify(); never looked up or run
        if RULES.get(_family(program)) is None and program not in self.extra_programs:
            raise PolicyDeniedError(
                f"Not allowed: `{program}` is not on the terminal allowlist.",
                hint="Add it to [terminal] extra_programs in config.toml if you trust it.",
            )
        found = shutil.which(first, path=self.search_path)
        if found is None:
            raise ToolInputError(
                f"`{first}` isn't installed (or not on Jarvis's PATH).",
                hint="Jarvis's PATH is set by [terminal] search_path.",
            )
        return found, program


def _program_name(name: str) -> str:
    """``python3.14`` -> ``python3``; ``pip3`` -> ``pip``."""
    if re.fullmatch(r"python3(\.\d+)?", name):
        return "python3"
    if re.fullmatch(r"pip3?(\.\d+)?", name):
        return "pip"
    return name


def _family(program: str) -> str:
    return "python" if program in {"python", "python3"} else program


# -- rules: (program, args) -> (risk, reason) ------------------------------------------------

Rule = Callable[[str, list[str]], tuple[RiskLevel, str]]


def _version_only(args: list[str]) -> bool:
    return bool(args) and all(a in {"--version", "-V", "version", "--help", "-h"} for a in args)


def _python(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if _version_only(args):
        return R.READ_ONLY, "prints the version"
    if not args:
        raise PolicyDeniedError("Not allowed: an interactive interpreter.")
    if args[0] in {"-c", "-e"} or any(a.startswith("-c") and len(a) > 2 for a in args[:1]):
        raise PolicyDeniedError("Not allowed: inline code (`-c`).", hint="Run a file instead.")
    if args[0] == "-m" and len(args) > 1:
        module, rest = args[1], args[2:]
        if module == "pytest":
            return R.LOW, "runs the tests"
        if module == "pip":
            return _pip("pip", rest)
        if module == "venv":
            return R.MEDIUM, "creates a virtual environment"
        return R.MEDIUM, f"runs the {module} module from this environment"
    if args[0].endswith(".py") or args[0].endswith(".ipynb"):
        return R.MEDIUM, "runs project code"
    raise PolicyDeniedError(f"Not allowed: `{program} {args[0]}` (run a .py file or -m module).")


def _pip(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    sub = args[0] if args else ""
    if _version_only(args) or sub in {"list", "show", "freeze", "check"}:
        return R.READ_ONLY, "lists installed packages"
    if sub in {"install", "uninstall"}:
        return R.HIGH, "downloads and installs third-party code"
    raise PolicyDeniedError(f"Not allowed: `pip {sub}`.")


def _git(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if args and args[0] in {"-c", "--exec-path", "-C", "--git-dir", "--work-tree"}:
        raise PolicyDeniedError("Not allowed: overriding git settings in the command.")
    sub, rest = (args[0], args[1:]) if args else ("", [])
    if _version_only(args):
        return R.READ_ONLY, "prints the version"
    inspect = {"status", "log", "diff", "show", "rev-parse", "describe", "blame", "ls-files"}
    if sub in inspect | {"shortlog", "grep", "reflog"} or (
        sub == "remote" and rest in ([], ["-v"])
    ):
        return R.READ_ONLY, "inspects the repository"
    if sub == "branch":
        if any(a in {"-d", "-D", "--delete", "-m", "-M"} for a in rest):
            return R.HIGH, "deletes or renames a branch"
        return (R.READ_ONLY, "lists branches") if not rest else (R.MEDIUM, "creates a branch")
    if sub == "fetch":
        return R.LOW, "downloads remote changes without changing your files"
    if sub in {"add", "commit", "stash", "switch", "pull", "merge", "tag"}:
        return R.MEDIUM, "changes the repository"
    if sub == "checkout":
        if "--" in rest or "." in rest:
            return R.HIGH, "discards local changes"
        return R.MEDIUM, "switches branch"
    if sub in {"push", "rebase", "clean", "restore"} or (sub == "reset" and "--hard" in rest):
        return R.HIGH, "can discard work or change the remote"
    if sub == "reset":
        return R.MEDIUM, "changes what is staged"
    if sub == "config":
        raise PolicyDeniedError("Not allowed: `git config` (it can make git run other programs).")
    raise PolicyDeniedError(f"Not allowed: `git {sub}`.")


def _js_package_manager(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    sub = args[0] if args else ""
    if _version_only(args) or sub in {"ls", "list", "outdated", "view", "why"}:
        return R.READ_ONLY, "inspects packages"
    if sub == "test" or (sub == "run" and len(args) > 1 and args[1] == "test"):
        return R.LOW, "runs the tests"
    if sub in {"run", "start", "dev", "serve", "build", "exec", "preview"}:
        return R.MEDIUM, "runs a project script"
    if sub in {"install", "i", "ci", "add", "uninstall", "remove", "update", "upgrade"}:
        return R.HIGH, "downloads and installs third-party code"
    raise PolicyDeniedError(f"Not allowed: `{program} {sub}`.")


def _npx(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    return R.HIGH, "may download and run a package"


def _js_runtime(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if _version_only(args):
        return R.READ_ONLY, "prints the version"
    if not args or args[0] in {"-e", "--eval", "-p", "--print"}:
        raise PolicyDeniedError("Not allowed: inline code or an interactive interpreter.")
    return R.MEDIUM, "runs project code"


def _pytest(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if "--collect-only" in args or _version_only(args):
        return R.READ_ONLY, "lists the tests"
    return R.LOW, "runs the tests"


def _make(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if args and args[0] in {"-n", "--dry-run"}:
        return R.READ_ONLY, "shows what make would do"
    target = args[0] if args else "default"
    return (R.LOW, "runs the tests") if target == "test" else (R.MEDIUM, f"runs make {target}")


def _cargo_go(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    sub = args[0] if args else ""
    if _version_only(args):
        return R.READ_ONLY, "prints the version"
    if sub in {"test", "check", "vet", "fmt", "clippy"}:
        return R.LOW, "checks or tests the code"
    if sub in {"run", "build"}:
        return R.MEDIUM, "builds or runs project code"
    if sub in {"install", "get", "add", "update"}:
        return R.HIGH, "downloads and installs third-party code"
    raise PolicyDeniedError(f"Not allowed: `{program} {sub}`.")


def _read_only(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    return R.READ_ONLY, "only reads information"


def _server(program: str, args: list[str]) -> tuple[RiskLevel, str]:
    if _version_only(args):
        return R.READ_ONLY, "prints the version"
    return R.MEDIUM, "starts a development server"


READ_ONLY_PROGRAMS = (
    "ls", "pwd", "which", "whoami", "date", "uname", "wc", "head", "tail", "du", "df",
    "sw_vers", "file", "stat", "tree",
)  # fmt: skip

RULES: dict[str, Rule] = {
    "python": _python,
    "pip": _pip,
    "git": _git,
    "npm": _js_package_manager,
    "yarn": _js_package_manager,
    "pnpm": _js_package_manager,
    "npx": _npx,
    "node": _js_runtime,
    "deno": _js_runtime,
    "bun": _js_runtime,
    "pytest": _pytest,
    "make": _make,
    "cargo": _cargo_go,
    "go": _cargo_go,
    "streamlit": _server,
    "uvicorn": _server,
    "flask": _server,
    "jupyter": _server,
    **dict.fromkeys(READ_ONLY_PROGRAMS, _read_only),
}

_SERVER_HINTS = (
    ("-m", "streamlit"), ("-m", "uvicorn"), ("-m", "flask"), ("-m", "http.server"),
    ("-m", "jupyter"), ("manage.py", "runserver"),
)  # fmt: skip


def looks_long_running(program: str, args: list[str]) -> bool:
    """Dev servers and watchers that never exit by themselves."""
    if program in {"streamlit", "uvicorn", "flask", "jupyter"} and not _version_only(args):
        return True
    if program in {"npm", "yarn", "pnpm"} and args:
        script = args[1] if args[0] == "run" and len(args) > 1 else args[0]
        return script in {"start", "dev", "serve", "preview", "watch"}
    joined = list(args)
    for first, second in _SERVER_HINTS:
        if first in joined and second in joined:
            return True
    return "--watch" in args or "--reload" in args


# -- diagnosis ---------------------------------------------------------------------------------

_DIAGNOSES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'"),
     "Missing Python package '{0}': install the project's dependencies "
     "(e.g. `.venv/bin/pip install -r requirements.txt`)."),
    (re.compile(r"Cannot find module '([^']+)'"),
     "Missing Node module '{0}': run `npm install` in the project."),
    (re.compile(r"npm ERR! [Mm]issing script: \"?([\w:-]+)"), "package.json has no '{0}' script."),
    (re.compile(r"(?:EADDRINUSE|[Aa]ddress already in use|[Pp]ort \d+ is already in use)"),
     "The port is already in use: stop the other server (terminal.list / terminal.stop)."),
    (re.compile(r"(?:command not found: (\S+)|(\S+): command not found)"),
     "'{0}' isn't installed."),
    (re.compile(r"=+ (\d+) failed"), "{0} test(s) failed; see the output above."),
    (re.compile(r"^SyntaxError: (.+)$", re.MULTILINE), "Syntax error in the code: {0}"),
    (re.compile(r"[Pp]ermission denied"), "Permission denied: a file or folder isn't accessible."),
    (re.compile(r"^(\w+(?:Error|Exception)): (.+)$", re.MULTILINE), "{0}: {1}"),
]  # fmt: skip


def diagnose(output: str) -> list[str]:
    """Short, actionable explanations of common failures (first match per kind)."""
    found: list[str] = []
    for pattern, template in _DIAGNOSES:
        matches = list(pattern.finditer(output))
        if not matches:
            continue
        groups = [g for g in matches[-1].groups() if g] or [""]
        text = template.format(*groups, *[""] * 3)
        if text not in found:
            found.append(text)
        if len(found) >= 2:
            break
    return found
