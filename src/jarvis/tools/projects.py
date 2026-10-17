"""Project tools: find a project by name, and work out how to run it.

``projects.inspect`` gathers the facts a small model would otherwise need many
file reads for: language and framework, virtualenv, entry points,
``package.json`` scripts, Makefile targets and commands from the README. From
those it suggests run, test and setup commands. It never runs anything. The
terminal tools do that, with their own policy and confirmation.

README text is untrusted, so commands taken from it are labelled as such, and
any later action needs confirmation (see taint in the dispatcher).
"""

from __future__ import annotations

import json
import re
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import FilesConfig
from jarvis.errors import JarvisError, ToolExecutionError
from jarvis.tools.base import PreparedAction, ToolSpec
from jarvis.tools.files import FileTools, PathText
from jarvis.tools.paths import PathPolicy, display

MARKERS: dict[str, str] = {
    "pyproject.toml": "Python",
    "setup.py": "Python",
    "requirements.txt": "Python",
    "Pipfile": "Python",
    "manage.py": "Python",
    "package.json": "Node.js",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
    "build.gradle": "Java",
    "Gemfile": "Ruby",
    "CMakeLists.txt": "C/C++",
    "Makefile": "Make",
    ".git": "",
}
FRAMEWORKS: dict[str, tuple[str, ...]] = {
    "Streamlit": ("streamlit",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi",),
    "Django": ("django",),
    "Jupyter": ("jupyter",),
    "PyTorch": ("torch",),
    "TensorFlow": ("tensorflow",),
    "scikit-learn": ("sklearn", "scikit-learn"),
}
PY_ENTRY_POINTS = ("main.py", "app.py", "streamlit_app.py", "run.py", "server.py", "manage.py")
VENV_DIRS = (".venv", "venv", "env")
_README_RUN = re.compile(
    r"^\s*(?:\$\s*)?((?:python3?|py|pip3?|uv|poetry|streamlit|uvicorn|flask|npm|npx|yarn|pnpm|"
    r"node|make|cargo|go|docker(?: compose)?|java|mvn|gradle|bundle|rails|jupyter|\./[\w.-]+)"
    r"\b[^\n`]{0,120})$",
    re.MULTILINE,
)
_MAX_README = 12_000
_FASTAPI_APP = re.compile(r"^\s*app\s*=\s*FastAPI\(", re.MULTILINE)


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100, description="project name or words in it")
    limit: int = Field(default=8, ge=1, le=20)


class InspectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: PathText = Field(description="project folder")


@dataclass
class ProjectFacts:
    path: Path
    kinds: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    python: str | None = None  # interpreter to use, relative to the project
    venv: str | None = None
    entry_points: list[str] = field(default_factory=list)
    npm_scripts: dict[str, str] = field(default_factory=dict)
    make_targets: list[str] = field(default_factory=list)
    readme_commands: list[str] = field(default_factory=list)
    run: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)


def _read_small(path: Path, limit: int = 64_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def inspect_project(path: Path) -> ProjectFacts:
    """Collect run facts for the project at ``path`` (reads a few small files)."""
    facts = ProjectFacts(path=path)
    names = {p.name for p in path.iterdir()}
    facts.kinds = _dedupe([MARKERS[n] for n in MARKERS if n in names])

    # Python ------------------------------------------------------------------------------
    for candidate in VENV_DIRS:
        python = path / candidate / "bin" / "python"
        if (path / candidate / "pyvenv.cfg").is_file() and python.exists():
            facts.venv = candidate
            facts.python = f"{candidate}/bin/python"
            break
    deps = ""
    if "requirements.txt" in names:
        deps += _read_small(path / "requirements.txt").lower()
    if "pyproject.toml" in names:
        try:
            pyproject = tomllib.loads(_read_small(path / "pyproject.toml"))
            project = pyproject.get("project", {})
            deps += " ".join(project.get("dependencies", [])).lower()
            for script in project.get("scripts", {}):
                facts.entry_points.append(f"script: {script}")
                if facts.venv and (path / facts.venv / "bin" / script).exists():
                    facts.run.append(f"{facts.venv}/bin/{script}")
        except tomllib.TOMLDecodeError:
            pass
    py_files = [n for n in PY_ENTRY_POINTS if n in names]
    sources = " ".join(_read_small(path / n, 20_000) for n in py_files).lower()
    for framework, needles in FRAMEWORKS.items():
        if any(n in deps or n in sources for n in needles):
            facts.frameworks.append(framework)
    if any(k == "Python" for k in facts.kinds) or py_files:
        if "Python" not in facts.kinds:
            facts.kinds.append("Python")
        py = facts.python or "python3"
        facts.entry_points += py_files
        if "Streamlit" in facts.frameworks:
            app = next((n for n in ("streamlit_app.py", "app.py", "main.py") if n in names), None)
            if app:
                facts.run.append(f"{py} -m streamlit run {app}")
        if "manage.py" in names:
            facts.run.append(f"{py} manage.py runserver")
        if "FastAPI" in facts.frameworks:
            for module in ("main", "app"):
                if _FASTAPI_APP.search(_read_small(path / f"{module}.py")):
                    facts.run.append(f"{py} -m uvicorn {module}:app --reload")
        facts.run += [f"{py} {n}" for n in py_files if n != "manage.py"]
        if (path / "tests").is_dir() or "pytest" in deps:
            facts.test.append(f"{py} -m pytest")
        if not facts.venv and ("requirements.txt" in names or "pyproject.toml" in names):
            facts.setup.append("python3 -m venv .venv")
            target = "-r requirements.txt" if "requirements.txt" in names else "-e ."
            facts.setup.append(f".venv/bin/pip install {target}")

    # Node.js -----------------------------------------------------------------------------
    if "package.json" in names:
        try:
            package = json.loads(_read_small(path / "package.json"))
            scripts = package.get("scripts", {})
            if isinstance(scripts, dict):
                facts.npm_scripts = {str(k): str(v)[:120] for k, v in list(scripts.items())[:20]}
        except json.JSONDecodeError:
            pass
        for script in ("dev", "start", "serve"):
            if script in facts.npm_scripts:
                facts.run.append("npm start" if script == "start" else f"npm run {script}")
        if "test" in facts.npm_scripts:
            facts.test.append("npm test")
        if "node_modules" not in names:
            facts.setup.append("npm install")

    # Make / Rust / Go ------------------------------------------------------------------------
    if "Makefile" in names:
        targets = re.findall(
            r"^([A-Za-z][\w-]*):(?!=)", _read_small(path / "Makefile"), re.MULTILINE
        )
        facts.make_targets = _dedupe(targets)[:20]
        facts.run += [
            f"make {t}" for t in ("run", "dev", "start", "serve") if t in facts.make_targets
        ]
        facts.test += ["make test"] if "test" in facts.make_targets else []
    if "Cargo.toml" in names:
        facts.run.append("cargo run")
        facts.test.append("cargo test")
    if "go.mod" in names:
        facts.run.append("go run .")
        facts.test.append("go test ./...")

    # README ------------------------------------------------------------------------------
    readme = next((path / n for n in sorted(names) if n.lower().startswith("readme")), None)
    if readme is not None and readme.is_file():
        text = _read_small(readme, _MAX_README)
        facts.readme_commands = _dedupe([m.strip() for m in _README_RUN.findall(text)])[:10]

    facts.run = _dedupe(facts.run)
    facts.test = _dedupe(facts.test)
    facts.setup = _dedupe(facts.setup)
    return facts


def summarise(facts: ProjectFacts, home: Path) -> str:
    lines = [f"Project {display(facts.path, home)}"]
    lines.append(f"  Type: {', '.join(facts.kinds) or 'unknown'}"
                 + (f" ({', '.join(facts.frameworks)})" if facts.frameworks else ""))  # fmt: skip
    if facts.kinds and "Python" in facts.kinds:
        lines.append(f"  Python: {facts.python or 'no virtualenv found (system python3)'}")
    if facts.entry_points:
        lines.append(f"  Entry points: {', '.join(facts.entry_points)}")
    if facts.npm_scripts:
        lines.append(
            "  npm scripts: " + ", ".join(f"{k} ({v})" for k, v in facts.npm_scripts.items())
        )
    if facts.make_targets:
        lines.append(f"  make targets: {', '.join(facts.make_targets)}")
    if facts.run:
        lines.append(f"  Suggested run command(s): {' | '.join(facts.run)}")
    if facts.test:
        lines.append(f"  Test command(s): {' | '.join(facts.test)}")
    if facts.setup:
        lines.append(f"  Setup (if dependencies are missing): {' && '.join(facts.setup)}")
    if facts.readme_commands:
        lines.append("  Commands in README (untrusted text): " + " | ".join(facts.readme_commands))
    return "\n".join(lines)


class ProjectTools:
    """Builds the ``projects.*`` tool specs."""

    def __init__(self, files: FileTools, paths: PathPolicy, config: FilesConfig) -> None:
        self._files = files
        self._paths = paths
        self._config = config

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "projects.find",
                "Find a code project folder by name (searches the project folders).",
                FindArgs,
                RiskLevel.READ_ONLY,
                self._prepare_find,
                idempotent=True,
            ),
            ToolSpec(
                "projects.inspect",
                "Show a project's type, entry points and suggested run/test commands.",
                InspectArgs,
                RiskLevel.READ_ONLY,
                self._prepare_inspect,
                idempotent=True,
            ),
        ]

    def _roots(self) -> list[Path]:
        roots = []
        for raw in self._config.project_roots:
            try:
                root = self._paths.readable(str(raw))
            except JarvisError:  # a root the policy refuses is simply skipped
                continue
            if root.is_dir():
                roots.append(root)
        return list(dict.fromkeys(roots))

    def _prepare_find(self, args: FindArgs) -> PreparedAction:
        roots = self._roots()
        home = self._paths.home

        def run(cancel: threading.Event) -> ToolResult:
            found, _source = self._files.find(args.name, roots, "folder", 50, max_depth=5)
            projects = [p for p in found if any((p / m).exists() for m in MARKERS)]
            # A folder inside another match is usually a package of that project.
            projects = [
                p for p in projects if not any(p != q and p.is_relative_to(q) for q in projects)
            ]
            projects = projects[: args.limit]
            entries = [
                {"path": display(p, home), "type": ", ".join(inspect_kinds(p)) or "unknown"}
                for p in projects
            ]
            if not entries:
                where = ", ".join(display(r, home) for r in roots[:4])
                message = f"No project named like '{args.name}' found in {where}…"
            elif len(entries) == 1:
                message = f"Found project {entries[0]['path']} ({entries[0]['type']})."
            else:
                message = f"{len(entries)} projects match '{args.name}':\n" + "\n".join(
                    f"  {e['path']} ({e['type']})" for e in entries
                )
            return ToolResult(
                tool="projects.find",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"projects": entries},
            )

        return PreparedAction(description=f"Find project '{args.name}'", run=run)

    def _prepare_inspect(self, args: InspectArgs) -> PreparedAction:
        path = self._paths.readable(args.path)
        if not path.is_dir():
            raise ToolExecutionError(f"{display(path, self._paths.home)} is not a folder.")

        def run(cancel: threading.Event) -> ToolResult:
            facts = inspect_project(path)
            return ToolResult(
                tool="projects.inspect",
                status=ToolStatus.SUCCESS,
                message=summarise(facts, self._paths.home),
                data={
                    "path": display(path, self._paths.home),
                    "run": facts.run,
                    "test": facts.test,
                    "setup": facts.setup,
                    "readme_commands": facts.readme_commands,
                    "python": facts.python,
                },
                untrusted=bool(facts.readme_commands),
            )

        return PreparedAction(description=f"Inspect {display(path, self._paths.home)}", run=run)


def inspect_kinds(path: Path) -> list[str]:
    try:
        names = {p.name for p in path.iterdir()}
    except OSError:
        return []
    return _dedupe([MARKERS[n] for n in MARKERS if n in names])
