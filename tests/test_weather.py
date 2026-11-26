"""weather.forecast with a fake Open-Meteo (no network)."""

from __future__ import annotations

import urllib.error
import urllib.parse
from collections.abc import Callable
from typing import Any

import pytest

from jarvis.api.dispatcher import ExecutionContext
from jarvis.api.schemas import ToolCall, ToolResult, ToolStatus
from jarvis.bootstrap import Runtime
from jarvis.tools import weather
from tests.fakes import ScriptedConfirmer

MakeRuntime = Callable[..., tuple[Runtime, ScriptedConfirmer]]

PLACES = {
    "London": [
        {"name": "London", "latitude": 51.5, "longitude": -0.13, "admin1": "England",
         "country": "United Kingdom", "country_code": "GB"},
        {"name": "London", "latitude": 42.98, "longitude": -81.23, "admin1": "Ontario",
         "country": "Canada", "country_code": "CA"},
    ],
    "Pune": [
        {"name": "Pune", "latitude": 18.52, "longitude": 73.86, "admin1": "Maharashtra",
         "country": "India", "country_code": "IN"},
    ],
}  # fmt: skip

FORECAST = {
    "current": {"temperature_2m": 28.6, "apparent_temperature": 31.2,
                "relative_humidity_2m": 62, "weather_code": 2, "wind_speed_10m": 11.4},
    "daily": {"time": ["2026-11-26", "2026-11-27", "2026-11-28"],
              "weather_code": [2, 61, 95], "temperature_2m_max": [31.4, 29.0, 27.5],
              "temperature_2m_min": [21.0, 20.6, 19.9],
              "precipitation_probability_max": [5, 60, 90]},
}  # fmt: skip


class FakeOpenMeteo:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.down = False

    def __call__(self, url: str, timeout: float) -> Any:
        assert timeout > 0
        self.urls.append(url)
        if self.down:
            raise urllib.error.URLError("nodename nor servname provided")
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        if url.startswith(weather.GEOCODING_URL):
            found = PLACES.get(query["name"])
            return {"results": found} if found else {"generationtime_ms": 0.4}
        assert url.startswith(weather.FORECAST_URL)
        return FORECAST


@pytest.fixture
def open_meteo(monkeypatch: pytest.MonkeyPatch) -> FakeOpenMeteo:
    fake = FakeOpenMeteo()
    monkeypatch.setattr(weather, "fetch_json", fake)
    return fake


def forecast(runtime: Runtime, context: ExecutionContext | None = None, **args: Any) -> ToolResult:
    return runtime.dispatcher.execute(ToolCall(tool="weather.forecast", arguments=args), context)


def test_forecast(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    runtime, confirmer = make_runtime(confirm=False)
    result = forecast(runtime, location="Pune")
    assert result.ok and confirmer.prompts == []  # LOW: no confirmation
    assert result.message == (
        "Weather for Pune, Maharashtra, India:\n"
        "Now 29°C and partly cloudy, feels like 31°C, humidity 62%, wind 11 km/h.\n"
        "Today: partly cloudy, high 31°C, low 21°C, 5% chance of rain.\n"
        "Tomorrow: light rain, high 29°C, low 21°C, 60% chance of rain.\n"
        "Saturday: thunderstorms, high 28°C, low 20°C, 90% chance of rain."
    )
    assert result.untrusted  # from an outside service
    forecast_query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(open_meteo.urls[1]).query))
    assert forecast_query["latitude"] == "18.52" and forecast_query["timezone"] == "auto"
    assert "temperature_unit" not in forecast_query


def test_only_the_place_name_is_sent(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    runtime, _ = make_runtime()
    forecast(runtime, location="London, Canada")
    geocode = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(open_meteo.urls[0]).query))
    assert geocode == {"name": "London", "count": "10", "language": "en"}
    assert "latitude=42.98" in open_meteo.urls[1]  # the qualifier picked Ontario
    forecast(runtime, location="london, canada")
    assert len(open_meteo.urls) == 3  # the place was looked up once


def test_home_location_and_units(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    runtime, _ = make_runtime(weather={"location": "Pune", "units": "fahrenheit"})
    result = forecast(runtime)
    assert result.ok and result.message.startswith("Weather for Pune")
    assert "°F" in result.message and "mph" in result.message
    assert "temperature_unit=fahrenheit" in open_meteo.urls[1]


def test_no_place_asks(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    runtime, _ = make_runtime()
    result = forecast(runtime)
    assert result.status is ToolStatus.NEEDS_CLARIFICATION
    assert result.message == "Which city should I get the weather for?"
    assert open_meteo.urls == []


def test_unknown_places_and_outages(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    runtime, _ = make_runtime()
    unknown = forecast(runtime, location="Atlantis")
    assert unknown.status is ToolStatus.FAILED and "Couldn't find a place called “Atlantis”" in (
        unknown.message
    )
    open_meteo.down = True
    outage = forecast(runtime, location="Pune")
    assert outage.status is ToolStatus.FAILED and outage.hint and "internet" in outage.hint


def test_asks_after_outside_content(make_runtime: MakeRuntime, open_meteo: FakeOpenMeteo) -> None:
    """Words from the request leave the Mac, so a tainted request asks first."""
    runtime, confirmer = make_runtime(confirm=False)
    result = forecast(runtime, ExecutionContext(tainted=True), location="Pune")
    assert result.status is ToolStatus.CANCELLED and len(confirmer.prompts) == 1
    assert open_meteo.urls == []


def test_fetching_is_limited_to_open_meteo() -> None:
    for url in (
        "file:///etc/passwd",
        "http://api.open-meteo.com/v1/forecast",
        "https://evil.test/",
    ):
        with pytest.raises(ValueError, match="not an Open-Meteo URL"):
            weather.fetch_json(url, 1)
