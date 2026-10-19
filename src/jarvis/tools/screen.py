"""Screen tools (opt-in): screenshot and on-device text recognition.

These tools are **not enabled by default**; add ``"screen.*"`` to
``policy.enabled_tools`` to use them. Even then:

* Capturing the screen is MEDIUM risk, so every capture asks first. Jarvis
  never watches the screen and never captures on its own schedule.
* Text recognition runs on-device (Apple Vision). A capture taken only to read
  text goes to a private temporary folder and is deleted as soon as it has
  been read.
* Recognised text is untrusted content, like a web page.
* Jarvis prefers app, file and browser tools (structured data) over screen
  text, and never clicks by screen coordinates.
"""

from __future__ import annotations

import tempfile
import threading
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.errors import ToolExecutionError, ToolInputError
from jarvis.platforms.base import PlatformAdapter, TextLine
from jarvis.tools.base import PreparedAction, ToolSpec
from jarvis.tools.files import PathText
from jarvis.tools.paths import PathPolicy, display

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".heic", ".tiff", ".tif", ".bmp", ".gif"})


class ReadTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: PathText | None = Field(
        default=None, description="image file to read; empty = capture the screen now"
    )
    max_chars: int = Field(default=3000, ge=200, le=20_000)


class CaptureArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: PathText | None = Field(
        default=None, description="where to save the PNG; default: Desktop"
    )


def format_lines(lines: list[TextLine], max_chars: int) -> tuple[str, bool]:
    """Join recognised lines; returns (text, truncated)."""
    text = "\n".join(line.text for line in lines if line.text.strip())
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


class ScreenTools:
    """Builds the ``screen.*`` tool specs."""

    def __init__(self, platform: PlatformAdapter, paths: PathPolicy) -> None:
        self._platform = platform
        self._paths = paths

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "screen.read_text",
                "Read the text on the screen (asks first) or in an image file. Prefer app, "
                "file and browser tools when they can answer instead.",
                ReadTextArgs,
                RiskLevel.MEDIUM,
                self._prepare_read_text,
            ),
            ToolSpec(
                "screen.capture",
                "Save a screenshot of the main display as a PNG file (asks first).",
                CaptureArgs,
                RiskLevel.MEDIUM,
                self._prepare_capture,
            ),
        ]

    # -- read text ------------------------------------------------------------------------

    def _recognise(self, image: Path, max_chars: int, source: str) -> ToolResult:
        lines = self._platform.recognize_text(image)
        text, truncated = format_lines(lines, max_chars)
        if not text:
            message = f"No text found in {source}."
        else:
            note = "\n[… more text not shown]" if truncated else ""
            message = f"Text in {source} ({len(lines)} line(s)):\n{text}{note}"
        return ToolResult(
            tool="screen.read_text",
            status=ToolStatus.SUCCESS,
            message=message,
            data={"lines": len(lines), "truncated": truncated, "source": source},
            untrusted=True,
        )

    def _prepare_read_text(self, args: ReadTextArgs) -> PreparedAction:
        if args.path is not None:
            image = self._paths.readable(args.path)
            shown = display(image, self._paths.home)
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ToolInputError(f"{shown} is not an image file.")
            if not image.is_file():
                raise ToolExecutionError(f"{shown} doesn't exist.")

            def read_file(cancel: threading.Event) -> ToolResult:
                return self._recognise(image, args.max_chars, shown)

            # Reading a file you pointed at is like files.read: no screen involved.
            return PreparedAction(
                description=f"Read the text in {shown}", run=read_file, risk=RiskLevel.READ_ONLY
            )

        def read_screen(cancel: threading.Event) -> ToolResult:
            with tempfile.TemporaryDirectory(prefix="jarvis-screen-") as folder:
                capture = Path(folder) / "screen.png"  # private folder, deleted below
                self._platform.capture_screen(capture)
                return self._recognise(capture, args.max_chars, "the screen")

        return PreparedAction(
            description="Capture the screen and read its text",
            run=read_screen,
            warning="The screenshot stays on this Mac and is deleted right after reading.",
        )

    # -- capture --------------------------------------------------------------------------

    def _default_path(self) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d at %H.%M.%S")
        return self._paths.home / "Desktop" / f"Jarvis screenshot {stamp}.png"

    def _prepare_capture(self, args: CaptureArgs) -> PreparedAction:
        if args.path is None:
            dest = self._paths.check_write(self._default_path().resolve())
        else:
            dest = self._paths.writable(args.path)
        if dest.suffix.lower() != ".png":
            raise ToolInputError("Screenshots are saved as .png files.")
        shown = display(dest, self._paths.home)
        if dest.exists():
            raise ToolExecutionError(f"{shown} already exists; Jarvis never overwrites files.")
        if not dest.parent.is_dir():
            raise ToolExecutionError(
                f"The folder {display(dest.parent, self._paths.home)} doesn't exist."
            )

        def run(cancel: threading.Event) -> ToolResult:
            self._platform.capture_screen(dest)
            ok = dest.is_file() and dest.stat().st_size > 0
            return ToolResult(
                tool="screen.capture",
                status=ToolStatus.SUCCESS if ok else ToolStatus.FAILED,
                message=f"Saved a screenshot to {shown}." if ok else "The screenshot wasn't saved.",
                data={"path": shown},
                verified=ok,
            )

        return PreparedAction(description=f"Capture the screen to {shown}", run=run)
