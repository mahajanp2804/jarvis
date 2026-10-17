from __future__ import annotations

import pytest

from jarvis.api.schemas import RiskLevel
from jarvis.errors import PolicyDeniedError, ToolInputError
from jarvis.tools.browser import check_typing_target, classify_click, format_snapshot, normalise_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "https://example.com"),
        ("https://example.com/a?b=1", "https://example.com/a?b=1"),
        ("localhost:8501", "http://localhost:8501"),
        ("http://127.0.0.1:3000/", "http://127.0.0.1:3000/"),
    ],
)
def test_normalise_url(raw: str, expected: str) -> None:
    assert normalise_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("file:///etc/passwd", PolicyDeniedError),
        ("javascript:alert(1)", PolicyDeniedError),
        ("chrome://settings", PolicyDeniedError),
        ("weather tomorrow", ToolInputError),
        ("", ToolInputError),
    ],
)
def test_refused_urls(raw: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        normalise_url(raw)


@pytest.mark.parametrize(
    ("element", "risk"),
    [
        ({"name": "Docs", "tag": "a"}, RiskLevel.LOW),
        ({"name": "Next page", "tag": "button"}, RiskLevel.LOW),
        ({"name": "Buy now", "tag": "button"}, RiskLevel.HIGH),
        ({"name": "Place order", "tag": "button"}, RiskLevel.HIGH),
        ({"name": "Send message", "tag": "button"}, RiskLevel.HIGH),
        ({"name": "Delete account", "tag": "a"}, RiskLevel.HIGH),
        ({"name": "Go", "tag": "button", "submits": True}, RiskLevel.HIGH),
    ],
)
def test_classify_click(element: dict[str, object], risk: RiskLevel) -> None:
    assert classify_click(element)[0] is risk


@pytest.mark.parametrize(
    ("element", "match"),
    [
        ({"tag": "input", "type": "password"}, "passwords"),
        ({"tag": "input", "autocomplete": "current-password"}, "passwords"),
        ({"tag": "input", "autocomplete": "cc-number"}, "payment"),
        ({"tag": "input", "field": "card_cvc"}, "payment"),
        ({"tag": "a", "ref": "e3"}, "not a text field"),
    ],
)
def test_refused_typing_targets(element: dict[str, object], match: str) -> None:
    with pytest.raises((PolicyDeniedError, ToolInputError), match=match):
        check_typing_target(element)


def test_format_snapshot_flags_captcha_and_login() -> None:
    text = format_snapshot(
        {
            "title": "T",
            "url": "https://x",
            "text": "hi",
            "text_length": 2,
            "elements": [],
            "captcha": True,
            "login": True,
        }
    )
    assert "CAPTCHA" in text and "log in themselves" in text
