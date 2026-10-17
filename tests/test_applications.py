from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.api.schemas import ToolCall, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.errors import AmbiguousAppError, AppNotFoundError
from jarvis.platforms.base import AppInfo
from jarvis.tools.applications import AppCatalog, normalise
from tests.fakes import DEFAULT_APPS, FakeAppControl, FakePlatform, ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]


def catalog(
    apps: list[AppInfo] | None = None,
    aliases: dict[str, str] | None = None,
    cache_seconds: float = 60,
    clock: Callable[[], float] | None = None,
) -> tuple[AppCatalog, FakeAppControl]:
    adapter = FakeAppControl(apps if apps is not None else DEFAULT_APPS)
    kwargs = {"clock": clock} if clock else {}
    cat = AppCatalog(
        adapter,
        [Path("/Applications")],
        aliases if aliases is not None else adapter.default_aliases(),
        cache_seconds,
        **kwargs,
    )
    return cat, adapter


def test_normalise() -> None:
    assert normalise("  Visual-Studio   Code! ") == "visual studio code"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Safari", "Safari"),
        ("safari", "Safari"),
        ("SAFARI", "Safari"),
        ("text edit", "TextEdit"),  # spacing differences
        ("code", "Visual Studio Code"),  # CFBundleName alt name
        ("vscode", "Visual Studio Code"),  # platform alias
        ("com.apple.Safari", "Safari"),  # bundle identifier
        ("calcu", "Calculator"),  # unique prefix
        ("booth", "Photo Booth"),  # unique whole word
        ("photos", "Photos"),  # exact beats prefix ("Photo Booth" also starts similarly)
    ],
)
def test_resolve(query: str, expected: str) -> None:
    cat, _ = catalog()
    assert cat.resolve(query).name == expected


def test_ambiguous_prefix_asks_for_clarification() -> None:
    cat, _ = catalog()
    with pytest.raises(AmbiguousAppError) as info:
        cat.resolve("photo")
    assert info.value.candidates == ["Photo Booth", "Photos"]


def test_duplicate_names_are_disambiguated_by_path() -> None:
    apps = [
        AppInfo("Foo", Path("/Applications/Foo.app"), "com.a.foo"),
        AppInfo("Foo", Path("/Users/me/Applications/Foo.app"), "com.b.foo"),
    ]
    cat, _ = catalog(apps)
    with pytest.raises(AmbiguousAppError) as info:
        cat.resolve("foo")
    assert all("(" in c for c in info.value.candidates)


def test_not_found_offers_suggestions_but_never_guesses() -> None:
    cat, _ = catalog()
    with pytest.raises(AppNotFoundError) as info:
        cat.resolve("safarii")
    assert info.value.suggestions == ["Safari"]
    assert info.value.hint and "Safari" in info.value.hint


def test_short_queries_do_not_partial_match() -> None:
    cat, _ = catalog()
    with pytest.raises(AppNotFoundError):
        cat.resolve("no")  # would otherwise prefix-match "Notes"


def test_alias_to_missing_app_falls_back_to_normal_matching() -> None:
    cat, _ = catalog()
    with pytest.raises(AppNotFoundError):
        cat.resolve("chrome")  # alias target "Google Chrome" is not installed


def test_user_alias() -> None:
    cat, _ = catalog(aliases={"browser": "Safari"})
    assert cat.resolve("Browser").name == "Safari"


def test_discovery_is_cached_until_expiry() -> None:
    now = [0.0]
    cat, adapter = catalog(cache_seconds=60, clock=lambda: now[0])
    cat.all()
    cat.all()
    assert adapter.discover_calls == 1
    now[0] = 61.0
    cat.all()
    assert adapter.discover_calls == 2
    cat.invalidate()
    cat.all()
    assert adapter.discover_calls == 3


# -- tools through the dispatcher -----------------------------------------------


def run(runtime: Runtime, tool: str, **arguments: object) -> tuple[ToolStatus, str, object]:
    result = runtime.dispatcher.execute(ToolCall(tool=tool, arguments=arguments))
    return result.status, result.message, result


def test_open_already_running(make_runtime: MakeRuntime, fake_platform: FakePlatform) -> None:
    fake_platform.apps.running.add("Safari")
    runtime, _ = make_runtime()
    status, message, _ = run(runtime, "apps.open", name="safari")
    assert status is ToolStatus.SUCCESS
    assert "already running" in message


def test_open_that_never_starts_fails_verification(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.launch_starts_app = False
    runtime, _ = make_runtime()
    status, message, result = run(runtime, "apps.open", name="Safari")
    assert status is ToolStatus.FAILED
    assert "not running after" in message
    assert result.verified is False  # type: ignore[attr-defined]
    assert result.attempts == 3  # type: ignore[attr-defined]  # retried: opening is idempotent
    assert fake_platform.apps.launched == ["Safari"] * 3


def test_open_unverifiable_is_reported_honestly(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.can_detect_running = False
    runtime, _ = make_runtime()
    status, message, result = run(runtime, "apps.open", name="Safari")
    assert status is ToolStatus.SUCCESS
    assert "could not verify" in message
    assert result.verified is None  # type: ignore[attr-defined]


def test_quit_that_hangs_reports_likely_dialog(
    make_runtime: MakeRuntime, fake_platform: FakePlatform
) -> None:
    fake_platform.apps.running.add("TextEdit")
    fake_platform.apps.quit_stops_app = False
    runtime, _ = make_runtime(confirm=True)
    status, message, result = run(runtime, "apps.quit", name="TextEdit")
    assert status is ToolStatus.FAILED
    assert "still running" in message
    assert "save" in (result.hint or "")  # type: ignore[attr-defined]


def test_ambiguous_request_needs_clarification(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    status, message, _ = run(runtime, "apps.open", name="photo")
    assert status is ToolStatus.NEEDS_CLARIFICATION
    assert "Photo Booth" in message


def test_list_filters_and_limits(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    status, message, result = run(runtime, "apps.list", query="photo")
    assert status is ToolStatus.SUCCESS
    assert result.data["apps"] == ["Photo Booth", "Photos"]  # type: ignore[attr-defined]
    _, message, result = run(runtime, "apps.list", limit=2)
    assert result.data["total"] == len(DEFAULT_APPS)  # type: ignore[attr-defined]
    assert len(result.data["apps"]) == 2  # type: ignore[attr-defined]
    assert "showing 2" in message


def test_list_hides_apps_jarvis_may_not_control(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime(policy={"app_allowlist": ["Safari", "Notes"]})
    _, _, result = run(runtime, "apps.list")
    assert result.data["apps"] == ["Notes", "Safari"]  # type: ignore[attr-defined]


def test_list_empty(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    status, message, _ = run(runtime, "apps.list", query="zzz")
    assert status is ToolStatus.SUCCESS
    assert message == "No applications found matching 'zzz'."


def test_app_without_bundle_id_resolves() -> None:
    cat, _ = catalog([AppInfo("Odd", Path("/Applications/Odd.app")), *DEFAULT_APPS])
    assert cat.resolve("odd").bundle_id is None


def test_list_multiword_query_falls_back_to_any_word(make_runtime: MakeRuntime) -> None:
    runtime, _ = make_runtime()
    _, _, result = run(runtime, "apps.list", query="photo edit")
    assert result.data["apps"] == ["Photo Booth", "Photos", "TextEdit"]  # type: ignore[attr-defined]
    _, _, result = run(runtime, "apps.list", query="photo booth")
    assert result.data["apps"] == ["Photo Booth"]  # type: ignore[attr-defined]
