"""Run fixed AppleScripts with ``osascript``, for the apps Jarvis controls.

* **No text from a request is put into a script.** Names, titles, message
  text and search words are passed as ``argv`` to fixed scripts (``on run
  argv``), so they can't change what a script does.
* **Automation permission**: the first script for an app shows macOS's
  "Jarvis wants to control <app>" prompt. Until it's allowed, scripts fail
  with a permission error that says where to turn it on. The timeout leaves
  time to answer that prompt.
* Results come back as text: one record per line, fields separated by a tab
  (see :func:`records`). Scripts replace tabs and line breaks inside values.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from jarvis.errors import OSPermissionError, ToolExecutionError

OSASCRIPT = "/usr/bin/osascript"
TIMEOUT_SECONDS = 30.0

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_NOT_ALLOWED = ("-1743", "not authorized to send apple events", "not allowed to send")

# AppleScript helper for scripts that return records: one value per field,
# with the separators replaced so a title can't break the format.
CLEAN = """\
on clean(v)
	if v is missing value then return ""
	set t to v as text
	set AppleScript's text item delimiters to {tab, return, linefeed}
	set parts to text items of t
	set AppleScript's text item delimiters to " "
	set t to parts as text
	set AppleScript's text item delimiters to ""
	return t
end clean"""


def records(text: str, fields: int) -> list[list[str]]:
    """Tab-separated lines -> lists of ``fields`` values (short lines are padded)."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("\t")
        rows.append([v.strip() for v in values[:fields]] + [""] * (fields - len(values)))
    return rows


class AppleScript:
    """Runs scripts that control one app, with clear errors."""

    def __init__(
        self, app: str, runner: Runner = subprocess.run, timeout: float = TIMEOUT_SECONDS
    ) -> None:
        self.app = app
        self._run = runner
        self._timeout = timeout

    def __call__(self, script: str, what: str, *args: str) -> str:
        try:
            proc = self._run(
                [OSASCRIPT, "-e", script, *args],
                capture_output=True, text=True, timeout=self._timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                f"{self.app} didn't answer while trying to {what}.",
                hint=f"If macOS asked whether Jarvis may control {self.app}, click OK and "
                "try again.",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"Couldn't {what}: {exc}") from exc
        if proc.returncode != 0:
            error = (proc.stderr or proc.stdout or "").strip()
            if any(marker in error.lower() for marker in _NOT_ALLOWED):
                raise OSPermissionError(
                    f"macOS doesn't allow Jarvis to control {self.app}.",
                    hint=f"System Settings > Privacy & Security > Automation: turn on "
                    f"{self.app} for the app that runs Jarvis (Terminal, or Python for "
                    "the service).",
                )
            raise ToolExecutionError(f"Couldn't {what}: {error or proc.returncode}")
        return proc.stdout or ""
