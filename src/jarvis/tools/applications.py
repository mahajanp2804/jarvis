"""Application tools: list, open and quit installed applications.

Name resolution is deliberately conservative. A request is matched in order
against aliases, exact names, then unique prefix/word matches. Several matches
produce a clarifying question, and a fuzzy guess is only ever offered as a
suggestion, never launched. Only applications found by discovery can be
launched, so a tool call can't be used to run an arbitrary executable path.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import ApplicationsConfig
from jarvis.errors import AmbiguousAppError, AppNotFoundError, PolicyDeniedError
from jarvis.platforms.base import AppControlAdapter, AppInfo
from jarvis.runtime.permissions import Decision, PolicyEngine
from jarvis.tools.base import PreparedAction, ToolSpec

_MIN_PARTIAL_QUERY = 3
_MAX_CANDIDATES = 8


class ListAppsArgs(BaseModel):
    """Arguments for ``apps.list``."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        max_length=100,
        description="one word the app name contains, e.g. 'photo'",
    )
    limit: int = Field(default=50, ge=1, le=500, description="Maximum names to return")


class AppNameArgs(BaseModel):
    """Arguments for ``apps.open`` and ``apps.quit``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120, description="application name, e.g. 'Safari'")

    @field_validator("name")
    @classmethod
    def _clean(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
            raise ValueError("must not contain control characters")
        return value


def normalise(text: str) -> str:
    """Case-fold and collapse punctuation/spaces: 'Visual-Studio  Code' -> 'visual studio code'."""
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _compact(text: str) -> str:
    return normalise(text).replace(" ", "")


class AppCatalog:
    """Cached view of installed applications plus name resolution."""

    def __init__(
        self,
        adapter: AppControlAdapter,
        search_paths: Sequence[Path],
        aliases: dict[str, str],
        cache_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._search_paths = list(search_paths)
        self._aliases = {normalise(k): v for k, v in aliases.items()}
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._apps: list[AppInfo] | None = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def all(self) -> list[AppInfo]:
        """Installed applications, rescanning disk at most every ``cache_seconds``."""
        with self._lock:
            now = self._clock()
            if self._apps is None or now - self._loaded_at > self._cache_seconds:
                self._apps = self._adapter.discover(self._search_paths)
                self._loaded_at = now
            return list(self._apps)

    def invalidate(self) -> None:
        with self._lock:
            self._apps = None

    def resolve(self, query: str) -> AppInfo:
        """Resolve a spoken/typed name to exactly one installed application.

        Raises:
            AmbiguousAppError: several applications match equally well.
            AppNotFoundError: nothing matches (with close-match suggestions).
        """
        apps = self.all()
        q = normalise(query)

        target = self._aliases.get(q)
        if target is not None:
            aliased = self._exact(apps, normalise(target), target)
            if len(aliased) == 1:
                return aliased[0]
            if len(aliased) > 1:
                raise AmbiguousAppError(query, _labels(aliased))
            # Alias target not installed: fall through and match the words as typed.

        exact = self._exact(apps, q, query)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise AmbiguousAppError(query, _labels(exact))

        if len(q) >= _MIN_PARTIAL_QUERY:
            partial = [
                app
                for app in apps
                if any(
                    normalise(n).startswith(q) or f" {q}" in f" {normalise(n)}"
                    for n in app.all_names
                )
            ]
            if len(partial) == 1:
                return partial[0]
            if len(partial) > 1:
                raise AmbiguousAppError(query, _labels(partial)[:_MAX_CANDIDATES])

        by_norm = {normalise(n): n for app in apps for n in app.all_names}
        close = difflib.get_close_matches(q, list(by_norm), n=3, cutoff=0.6)
        raise AppNotFoundError(query, [by_norm[c] for c in close])

    @staticmethod
    def _exact(apps: Sequence[AppInfo], q: str, raw: str) -> list[AppInfo]:
        compact = q.replace(" ", "")
        return [
            app
            for app in apps
            if any(normalise(n) == q or _compact(n) == compact for n in app.all_names)
            or (app.bundle_id is not None and app.bundle_id.casefold() == raw.strip().casefold())
        ]


def _filter_by_query(apps: Sequence[AppInfo], query: str) -> list[AppInfo]:
    """Apps whose names contain ``query``; if none, those containing any of its words."""
    q = normalise(query)
    matches = [a for a in apps if any(q in normalise(n) for n in a.all_names)]
    words = [w for w in q.split() if len(w) >= _MIN_PARTIAL_QUERY]
    if matches or len(words) < 2:
        return matches
    return [a for a in apps if any(w in normalise(n) for n in a.all_names for w in words)]


def _labels(apps: Sequence[AppInfo]) -> list[str]:
    """Display names, disambiguated with the path when two share a name."""
    names = [a.name for a in apps]
    return sorted(
        (f"{a.name} ({a.path})" if names.count(a.name) > 1 else a.name for a in apps),
        key=str.casefold,
    )


class _Wait(Enum):
    CONFIRMED = "confirmed"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


class ApplicationTools:
    """Builds the ``apps.*`` tool specs around a platform adapter."""

    def __init__(
        self,
        adapter: AppControlAdapter,
        catalog: AppCatalog,
        policy: PolicyEngine,
        config: ApplicationsConfig,
        *,
        poll_interval: float = 0.1,
    ) -> None:
        self._adapter = adapter
        self._catalog = catalog
        self._policy = policy
        self._config = config
        self._poll_interval = poll_interval

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="apps.list",
                description="List installed applications Jarvis may control, optionally filtered.",
                args_model=ListAppsArgs,
                risk=RiskLevel.READ_ONLY,
                prepare=self._prepare_list,
                idempotent=True,
            ),
            ToolSpec(
                name="apps.open",
                description="Open an installed application by name, or bring it to the front.",
                args_model=AppNameArgs,
                risk=RiskLevel.LOW,
                prepare=self._prepare_open,
                idempotent=True,  # opening an open app just brings it to the front
            ),
            ToolSpec(
                name="apps.quit",
                description="Ask a running application to quit normally (it may prompt to save).",
                args_model=AppNameArgs,
                risk=RiskLevel.MEDIUM,
                prepare=self._prepare_quit,
            ),
        ]

    # -- helpers -----------------------------------------------------------------

    def _permitted(self, app: AppInfo) -> bool:
        return self._policy.check_app(app.name, app.bundle_id).decision is not Decision.DENY

    def _resolve_permitted(self, name: str) -> AppInfo:
        app = self._catalog.resolve(name)
        verdict = self._policy.check_app(app.name, app.bundle_id)
        if verdict.decision is Decision.DENY:
            raise PolicyDeniedError(
                f"Not allowed: {verdict.reason}.",
                hint="Change [policy] in config.toml if this is intended.",
            )
        return app

    def _wait_until(
        self, predicate: Callable[[], bool | None], timeout: float, cancel: threading.Event
    ) -> _Wait:
        deadline = time.monotonic() + timeout
        while True:
            state = predicate()
            if state is None:
                return _Wait.UNKNOWN
            if state:
                return _Wait.CONFIRMED
            if time.monotonic() >= deadline:
                return _Wait.TIMED_OUT
            if cancel.wait(self._poll_interval):
                return _Wait.CANCELLED

    # -- apps.list ---------------------------------------------------------------

    def _prepare_list(self, args: ListAppsArgs) -> PreparedAction:
        suffix = f" matching '{args.query}'" if args.query else ""

        def run(cancel: threading.Event) -> ToolResult:
            apps = [a for a in self._catalog.all() if self._permitted(a)]
            if args.query:
                apps = _filter_by_query(apps, args.query)
            names = sorted({a.name for a in apps}, key=str.casefold)
            shown = names[: args.limit]
            if not names:
                message = f"No applications found{suffix}."
            else:
                more = f" (showing {len(shown)})" if len(shown) < len(names) else ""
                message = f"Found {len(names)} application(s){suffix}{more}:\n" + "\n".join(
                    f"  {n}" for n in shown
                )
            return ToolResult(
                tool="apps.list",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"apps": shown, "total": len(names)},
            )

        return PreparedAction(description=f"List installed applications{suffix}", run=run)

    # -- apps.open ---------------------------------------------------------------

    def _prepare_open(self, args: AppNameArgs) -> PreparedAction:
        app = self._resolve_permitted(args.name)
        timeout = self._config.launch_timeout_seconds

        def run(cancel: threading.Event) -> ToolResult:
            was_running = self._adapter.is_running(app)
            self._adapter.launch(app)
            outcome = self._wait_until(lambda: self._adapter.is_running(app), timeout, cancel)
            data = {"app": app.name, "path": str(app.path), "bundle_id": app.bundle_id}
            if outcome is _Wait.CONFIRMED:
                message = (
                    f"{app.name} was already running; brought it to the front."
                    if was_running
                    else f"Opened {app.name}."
                )
                return ToolResult(
                    tool="apps.open",
                    status=ToolStatus.SUCCESS,
                    message=message,
                    data=data,
                    verified=True,
                )
            if outcome is _Wait.UNKNOWN:
                return ToolResult(
                    tool="apps.open",
                    status=ToolStatus.SUCCESS,
                    message=f"Asked the system to open {app.name} (could not verify it started).",
                    data=data,
                    verified=None,
                )
            if outcome is _Wait.CANCELLED:
                return ToolResult(
                    tool="apps.open",
                    status=ToolStatus.CANCELLED,
                    message=f"Stopped waiting for {app.name} to start.",
                    data=data,
                )
            return ToolResult(
                tool="apps.open",
                status=ToolStatus.FAILED,
                message=f"The system accepted the request, but {app.name} was not running "
                f"after {timeout:g}s.",
                hint="The app may have crashed on launch or be blocked by a system dialog.",
                data=data,
                verified=False,
                retryable=True,
            )

        return PreparedAction(description=f"Open {app.name} ({app.path})", run=run)

    # -- apps.quit ---------------------------------------------------------------

    def _prepare_quit(self, args: AppNameArgs) -> PreparedAction:
        app = self._resolve_permitted(args.name)
        timeout = self._config.quit_timeout_seconds
        data = {"app": app.name, "path": str(app.path), "bundle_id": app.bundle_id}

        if self._adapter.is_running(app) is False:

            def noop(cancel: threading.Event) -> ToolResult:
                return ToolResult(
                    tool="apps.quit",
                    status=ToolStatus.SUCCESS,
                    message=f"{app.name} isn't running.",
                    data=data,
                    verified=True,
                )

            return PreparedAction(
                description=f"Nothing to do: {app.name} isn't running", run=noop, noop=True
            )

        def is_stopped() -> bool | None:
            running = self._adapter.is_running(app)
            return None if running is None else not running

        def run(cancel: threading.Event) -> ToolResult:
            self._adapter.quit(app)
            outcome = self._wait_until(is_stopped, timeout, cancel)
            if outcome is _Wait.CONFIRMED:
                return ToolResult(
                    tool="apps.quit",
                    status=ToolStatus.SUCCESS,
                    message=f"Quit {app.name}.",
                    data=data,
                    verified=True,
                )
            if outcome is _Wait.UNKNOWN:
                return ToolResult(
                    tool="apps.quit",
                    status=ToolStatus.SUCCESS,
                    message=f"Asked {app.name} to quit (could not verify it stopped).",
                    data=data,
                    verified=None,
                )
            if outcome is _Wait.CANCELLED:
                return ToolResult(
                    tool="apps.quit",
                    status=ToolStatus.CANCELLED,
                    message=f"Stopped waiting for {app.name} to quit.",
                    data=data,
                )
            return ToolResult(
                tool="apps.quit",
                status=ToolStatus.FAILED,
                message=f"Asked {app.name} to quit, but it is still running after {timeout:g}s.",
                hint="It is probably waiting for you to save or confirm something.",
                data=data,
                verified=False,
            )

        return PreparedAction(
            description=f"Quit {app.name}",
            run=run,
            warning="The app may ask you to save unsaved work.",
        )
