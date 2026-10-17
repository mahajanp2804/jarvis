from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis.errors import PolicyDeniedError, ToolInputError
from jarvis.tools.paths import PathPolicy, display, match_score, name_tokens


@pytest.fixture
def policy(home: Path) -> PathPolicy:
    return PathPolicy(
        [Path("~")],
        [Path("~/Desktop"), Path("~/Downloads")],
        [Path("~/Library")],
        home=home,
    )


def test_name_tokens() -> None:
    assert name_tokens("StockPrice-predictor_v2") == ["stock", "price", "predictor", "v2"]
    assert name_tokens("HTMLParser.py") == ["html", "parser", "py"]


@pytest.mark.parametrize(
    ("query", "name", "score"),
    [
        ("stock price predictor", "Stock Price Predictor", 3),
        ("stock price predictor", "stock-price-predictor", 3),
        ("stock price predictor", "StockPricePredictor", 3),
        ("stock predictor", "StockPricePredictor", 2),
        ("pric", "StockPricePredictor", 2),
        ("ockpri", "StockPricePredictor", 1),
        ("weather", "StockPricePredictor", 0),
    ],
)
def test_match_score(query: str, name: str, score: int) -> None:
    assert match_score(query, name) == score


def test_display(home: Path) -> None:
    assert display(home / "Desktop" / "x.txt", home) == "~/Desktop/x.txt"
    assert display(Path("/etc/hosts"), home) == "/etc/hosts"


def test_reads_inside_home_writes_only_in_write_roots(policy: PathPolicy, home: Path) -> None:
    assert policy.readable("~/Pictures/cat.jpg") == home / "Pictures" / "cat.jpg"
    assert policy.writable("~/Desktop/notes.txt") == home / "Desktop" / "notes.txt"
    assert policy.readable("Desktop/notes.txt") == home / "Desktop" / "notes.txt"  # relative to ~
    with pytest.raises(PolicyDeniedError, match="only change files in"):
        policy.writable("~/Pictures/cat.jpg")


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("/etc/passwd", "outside the folders"),
        ("~/../../etc/passwd", "outside the folders"),
        ("~/Library/Preferences/x.plist", "protected location"),
        ("~/.ssh/id_rsa", "hidden|key"),
        ("~/.ssh/config", "hidden"),
        ("~/Desktop/project/.env", "hidden"),
        ("~/Desktop/deploy.pem", "key or credential"),
        ("~/Desktop/id_ed25519", "key or credential"),
    ],
)
def test_refused_paths(policy: PathPolicy, raw: str, match: str) -> None:
    with pytest.raises(PolicyDeniedError, match=match):
        policy.readable(raw)


def test_symlinks_cannot_escape(policy: PathPolicy, home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / "Desktop" / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PolicyDeniedError, match="outside"):
        policy.readable("~/Desktop/link/secret.txt")
    (home / "Desktop" / "to-library").symlink_to(home / "Library", target_is_directory=True)
    with pytest.raises(PolicyDeniedError, match="protected"):
        policy.writable("~/Desktop/to-library/x")


@pytest.mark.parametrize("raw", ["", "   ", "~root/x", "a\x00b"])
def test_invalid_input(policy: PathPolicy, raw: str) -> None:
    with pytest.raises(ToolInputError):
        policy.resolve(raw)


def test_hidden_allowed_when_configured(home: Path) -> None:
    lenient = PathPolicy([Path("~")], [Path("~/Desktop")], allow_hidden=True, home=home)
    assert lenient.readable("~/Desktop/.env").name == ".env"
    with pytest.raises(PolicyDeniedError, match="key"):
        lenient.readable("~/.ssh/id_rsa")  # key names stay refused


def test_home_is_resolved_like_macos_tmp(home: Path) -> None:
    # /var -> /private/var on macOS: roots and paths must be compared resolved.
    assert os.path.realpath(home) == str(home)
