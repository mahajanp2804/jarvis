"""Append-only audit log of every action Jarvis considered, as JSON lines.

Each record says what was requested, what the policy decided, and what
happened. Arguments are redacted before writing. The file lives in the log
directory (``~/Library/Logs/Jarvis/audit.jsonl`` on macOS), is created with
0600 permissions, and rotates at 5 MB.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.logging_setup import redact, redact_data, secure_file_handler


class AuditLog:
    """Writes structured audit records to a dedicated rotating file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # A private logger instance (not registered globally) so records never
        # leak into the main application log or other handlers.
        self._logger = logging.Logger(f"jarvis.audit.{id(self)}", level=logging.INFO)
        handler = secure_file_handler(path, max_bytes=5_000_000, backups=5)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._handler = handler

    def record(
        self,
        *,
        request_id: str,
        tool: str,
        arguments: dict[str, Any],
        decision: str,
        status: str,
        message: str,
        dry_run: bool,
        duration_ms: float | None = None,
        verified: bool | None = None,
    ) -> None:
        """Append one audit record."""
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "tool": tool,
            "arguments": redact_data(arguments),
            "decision": decision,
            "status": status,
            "verified": verified,
            "dry_run": dry_run,
            "duration_ms": None if duration_ms is None else round(duration_ms, 1),
            "message": redact(message),
        }
        self._logger.info(json.dumps(entry, ensure_ascii=False, default=str))

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._logger.removeHandler(self._handler)
        self._handler.close()
