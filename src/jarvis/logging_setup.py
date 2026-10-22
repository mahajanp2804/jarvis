"""Logging configuration with secret redaction.

All handlers share :class:`RedactingFilter`, so credentials that slip into a log
message (an API key in an error string, a bearer token in a URL) are masked
before they reach disk. Log files are created owner-readable only (0600).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "jarvis"
REDACTED = "***"

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|access[_-]?key|secret|token|passw(or)?d|pwd|authorization|cookie|session[_-]?id)",
    re.IGNORECASE,
)

# Words that name a secret even without a "key=value" shape ("my PIN is 1234").
_SECRET_WORDS = re.compile(
    r"\b(?:passw(?:or)?ds?|passcodes?|passphrases?|pin(?:\s+(?:code|number))?|otp|"
    r"one[- ]time (?:code|password)|2fa|mfa|cvv|cvc|credit[- ]?card|card number|"
    r"debit card|iban|routing number|account number|social security|ssn|passport number|"
    r"api[- _]?keys?|secret(?: key)?s?|tokens?|private keys?|seed phrases?|"
    r"recovery (?:codes?|phrases?|keys?))\b",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # key=value / key: value pairs whose key looks sensitive
    (
        re.compile(
            r"(?i)\b([\w-]*(?:api[_-]?key|access[_-]?key|secret|token|passw(?:or)?d|pwd)[\w-]*)"
            r"(\s*[:=]\s*)(['\"]?)[^\s'\",;&]+"
        ),
        rf"\1\2\3{REDACTED}",
    ),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), rf"\1 {REDACTED}"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), REDACTED),  # OpenAI / Anthropic style
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),  # GitHub tokens
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), REDACTED),  # Hugging Face tokens
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),  # AWS access key IDs
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), REDACTED),  # Slack tokens
    (re.compile(r"://([^/\s:@]+):([^/\s@]+)@"), rf"://\1:{REDACTED}@"),  # URL credentials
]


def redact(text: str) -> str:
    """Mask substrings of ``text`` that look like credentials."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mentions_secret(text: str) -> bool:
    """True if ``text`` names a kind of secret (password, PIN, card number, token...)."""
    return bool(_SECRET_WORDS.search(text))


def redact_data(value: Any) -> Any:
    """Recursively redact a JSON-like structure (dicts, lists, strings).

    Values under sensitive keys are masked, and so is the ``value`` of a
    ``{"name": ..., "value": ...}`` pair whose name looks sensitive.
    """
    if isinstance(value, Mapping):
        name = value.get("name")
        secret_pair = isinstance(name, str) and (
            bool(_SENSITIVE_KEY.search(name)) or mentions_secret(name)
        )
        return {
            k: (
                REDACTED
                if isinstance(k, str)
                and (_SENSITIVE_KEY.search(k) or (secret_pair and k == "value"))
                else redact_data(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_data(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    return value


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the fully formatted message and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


class BriefFormatter(logging.Formatter):
    """Console formatter that omits tracebacks (they go to the log file only)."""

    def format(self, record: logging.LogRecord) -> str:
        saved = (record.exc_info, record.exc_text)
        record.exc_info, record.exc_text = None, None
        try:
            text = super().format(record)
        finally:
            record.exc_info, record.exc_text = saved
        return f"{text} (traceback in log file)" if saved[0] else text


def secure_file_handler(path: Path, *, max_bytes: int, backups: int) -> RotatingFileHandler:
    """Create a rotating file handler whose directory is 0700 and file is 0600."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        # Create with restrictive permissions before the handler opens it.
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        os.close(fd)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.addFilter(RedactingFilter())
    return handler


def configure_logging(level: str, log_dir: Path, *, console: bool = True) -> logging.Logger:
    """Configure the ``jarvis`` logger hierarchy. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = secure_file_handler(log_dir / "jarvis.log", max_bytes=1_000_000, backups=3)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        # The CLI prints user-facing results itself; the console only shows problems.
        console_handler.setLevel(max(logging.WARNING, logging.getLevelName(level)))
        console_handler.setFormatter(BriefFormatter("jarvis: %(levelname)s: %(message)s"))
        console_handler.addFilter(RedactingFilter())
        logger.addHandler(console_handler)
    return logger
