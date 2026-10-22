"""Exception hierarchy for expected, user-reportable failures.

Anything derived from :class:`JarvisError` is an anticipated failure that can be
shown to the user as-is (with an optional actionable ``hint``). Any other
exception reaching the dispatcher is treated as an internal bug: it is logged
with a traceback and reported to the user generically.
"""

from __future__ import annotations

from collections.abc import Sequence


class JarvisError(Exception):
    """Base class for errors that are safe and useful to show to the user."""

    def __init__(self, message: str, *, hint: str | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        # True for transient failures (timeouts, a page still loading) that an
        # idempotent tool may safely retry. See Dispatcher.
        self.retryable = retryable

    def __str__(self) -> str:
        return self.message


class ConfigError(JarvisError):
    """The configuration file or environment overrides are invalid."""


class ToolInputError(JarvisError):
    """Arguments for a tool failed validation."""


class UnknownToolError(JarvisError):
    """A tool call referenced a tool that is not registered."""


class PolicyDeniedError(JarvisError):
    """The permission policy forbids this action."""


class AppNotFoundError(JarvisError):
    """No installed application matches the requested name."""

    def __init__(self, query: str, suggestions: Sequence[str] = ()) -> None:
        self.query = query
        self.suggestions = list(suggestions)
        hint = None
        if self.suggestions:
            hint = "Did you mean: " + ", ".join(self.suggestions) + "?"
        super().__init__(f"No installed application matches '{query}'.", hint=hint)


class AmbiguousAppError(JarvisError):
    """More than one installed application matches; the user must choose."""

    def __init__(self, query: str, candidates: Sequence[str]) -> None:
        self.query = query
        self.candidates = list(candidates)
        super().__init__(
            f"'{query}' matches several applications: {', '.join(self.candidates)}.",
            hint="Say the full application name.",
        )


class AppNameUnclearError(AmbiguousAppError):
    """No application has this name, but one or two are near-certain matches.

    Usually a misheard or mistyped name ("photo wood" for Photo Booth). Jarvis
    asks instead of guessing, and doesn't hand the name to the AI model.
    """

    def __init__(self, query: str, suggestions: Sequence[str]) -> None:
        self.query = query
        self.candidates = list(suggestions)
        names = " or ".join(self.candidates)
        JarvisError.__init__(self, f"There's no app called '{query}'. Did you mean {names}?")


class OSPermissionError(JarvisError):
    """The operating system blocked the action (e.g. a missing privacy permission)."""


class UnsupportedPlatformError(JarvisError):
    """The feature is not implemented on this operating system."""


class ToolExecutionError(JarvisError):
    """A tool ran but the underlying operation failed."""
