from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import pytest

from jarvis.logging_setup import REDACTED, configure_logging, redact, redact_data
from jarvis.runtime.audit import AuditLog


@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("api_key=abcd1234efgh5678", "abcd1234efgh5678"),
        ('OPENAI_API_KEY: "sk-proj-abcdefghijklmnopqrstuv"', "sk-proj-abcdefghijklmnopqrstuv"),
        ("password=hunter2!", "hunter2!"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload", "eyJhbGciOiJIUzI1NiJ9"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
        ("using hf_abcdefghijklmnopqrstuvwxyz", "hf_abcdefghijklmnopqrstuvwxyz"),
        ("key AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
        ("https://user:s3cret@example.com/x", "s3cret"),
    ],
)
def test_redact_masks_secrets(raw: str, secret: str) -> None:
    out = redact(raw)
    assert secret not in out
    assert REDACTED in out


def test_redact_leaves_ordinary_text() -> None:
    text = "Opened Safari (/Applications/Safari.app) in 120ms"
    assert redact(text) == text


def test_redact_data_masks_sensitive_keys_recursively() -> None:
    data = {"name": "Safari", "nested": {"password": "x", "items": ["token=abc123456"]}}
    out = redact_data(data)
    assert out["name"] == "Safari"
    assert out["nested"]["password"] == REDACTED
    assert "abc123456" not in out["nested"]["items"][0]


def test_log_file_is_private_and_redacted(tmp_path: Path) -> None:
    logger = configure_logging("DEBUG", tmp_path / "logs", console=False)
    logger.info("calling api with api_key=%s", "supersecretvalue")
    for handler in logger.handlers:
        handler.flush()
    log_file = tmp_path / "logs" / "jarvis.log"
    content = log_file.read_text(encoding="utf-8")
    assert "supersecretvalue" not in content
    assert "api_key=***" in content
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "logs").stat().st_mode) == 0o700
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_exception_text_is_redacted(tmp_path: Path) -> None:
    logger = configure_logging("INFO", tmp_path, console=False)
    try:
        raise RuntimeError("failed with token=abcdef123456")
    except RuntimeError:
        logger.exception("boom")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "jarvis.log").read_text(encoding="utf-8")
    assert "abcdef123456" not in content
    assert "RuntimeError" in content
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_audit_log_writes_redacted_json_lines(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.record(
        request_id="r1",
        tool="apps.open",
        arguments={"name": "Safari", "api_key": "k"},
        decision="allow",
        status="success",
        message="Opened Safari with password=pw123",
        dry_run=False,
        duration_ms=12.345,
        verified=True,
    )
    audit.close()
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tool"] == "apps.open"
    assert entry["arguments"] == {"name": "Safari", "api_key": REDACTED}
    assert "pw123" not in entry["message"]
    assert entry["duration_ms"] == 12.3
    assert entry["verified"] is True
    assert stat.S_IMODE((tmp_path / "audit.jsonl").stat().st_mode) == 0o600


def test_audit_records_do_not_reach_root_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    with caplog.at_level(logging.DEBUG):
        audit.record(
            request_id="r",
            tool="t",
            arguments={},
            decision="allow",
            status="success",
            message="m",
            dry_run=True,
        )
    audit.close()
    assert caplog.records == []
