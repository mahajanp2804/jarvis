"""Weather: current conditions and a three-day forecast from Open-Meteo.

* **No account or API key.** Open-Meteo (open-meteo.com) is free for
  non-commercial use. Jarvis sends it the place name (to find coordinates) and
  then the coordinates, nothing else.
* LOW risk, like a web search: nothing on the Mac changes, but words from the
  request leave the Mac. So after a request has read outside content, it asks.
* With no place in the request, ``[weather] location`` is used; if that's
  empty too, Jarvis asks which place.
* The forecast is outside content: shown to the model as data.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jarvis import __version__
from jarvis.api.schemas import RiskLevel, ToolResult, ToolStatus
from jarvis.config import WeatherConfig
from jarvis.errors import NeedsClarificationError, ToolExecutionError
from jarvis.tools.base import PreparedAction, ToolSpec

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes, as Open-Meteo documents them.
CONDITIONS = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}  # fmt: skip

FetchJSON = Callable[[str, float], Any]


def fetch_json(url: str, timeout: float) -> Any:
    if not url.startswith(("https://geocoding-api.open-meteo.com/", "https://api.open-meteo.com/")):
        raise ValueError(f"not an Open-Meteo URL: {url}")
    headers = {"User-Agent": f"jarvis/{__version__}"}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed https hosts
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


class ForecastArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str | None = Field(
        default=None, min_length=1, max_length=100, description="city or place; omit for home"
    )


def _place_name(place: dict[str, Any]) -> str:
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    seen: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(str(part))
    return ", ".join(seen)


def pick_place(results: list[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """The best geocoding result for "Springfield, Illinois" or "Pune"."""
    qualifiers = [q.strip().casefold() for q in wanted.split(",")[1:] if q.strip()]
    for place in results:
        where = " ".join(
            str(place.get(k, "")) for k in ("admin1", "country", "country_code")
        ).casefold()
        if all(q in where for q in qualifiers):
            return place
    return None


def describe_forecast(place: str, data: dict[str, Any], units: str) -> str:
    degree = "°F" if units == "fahrenheit" else "°C"
    speed = "mph" if units == "fahrenheit" else "km/h"
    now = data.get("current", {})
    lines = [f"Weather for {place}:"]
    if now:
        condition = CONDITIONS.get(now.get("weather_code", -1), "unknown conditions")
        lines.append(
            f"Now {round(now['temperature_2m'])}{degree} and {condition}, feels like "
            f"{round(now['apparent_temperature'])}{degree}, humidity "
            f"{now['relative_humidity_2m']}%, wind {round(now['wind_speed_10m'])} {speed}."
        )
    daily = data.get("daily", {})
    for index, day in enumerate(daily.get("time", [])):
        label = (
            ("Today", "Tomorrow")[index] if index < 2 else date.fromisoformat(day).strftime("%A")
        )
        condition = CONDITIONS.get(daily["weather_code"][index], "unknown conditions")
        rain = daily.get("precipitation_probability_max", [None] * (index + 1))[index]
        lines.append(
            f"{label}: {condition}, high {round(daily['temperature_2m_max'][index])}{degree}, "
            f"low {round(daily['temperature_2m_min'][index])}{degree}"
            + (f", {rain}% chance of rain." if rain is not None else ".")
        )
    return "\n".join(lines)


class WeatherTools:
    """Builds the ``weather.*`` tool specs."""

    def __init__(self, config: WeatherConfig, fetch: FetchJSON | None = None) -> None:
        self._config = config
        self._fetch = fetch  # None: the module's fetch_json, looked up at call time
        self._places: dict[str, dict[str, Any]] = {}  # geocoding cache for this run

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "weather.forecast",
                "Get the current weather and a 3-day forecast for a place.",
                ForecastArgs,
                RiskLevel.LOW,
                self._prepare_forecast,
                idempotent=True,
            )
        ]

    def _get(self, url: str, what: str) -> Any:
        try:
            return (self._fetch or fetch_json)(url, self._config.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise ToolExecutionError(
                f"Couldn't {what}: {getattr(exc, 'reason', exc)}",
                hint="Check the internet connection; the weather comes from open-meteo.com.",
                retryable=True,
            ) from exc

    def _geocode(self, wanted: str) -> dict[str, Any]:
        key = wanted.casefold()
        if key not in self._places:
            name = wanted.split(",")[0].strip()
            query = urllib.parse.urlencode({"name": name, "count": 10, "language": "en"})
            found = self._get(f"{GEOCODING_URL}?{query}", f"look up “{wanted}”")
            place = pick_place(found.get("results") or [], wanted)
            if place is None:
                raise ToolExecutionError(
                    f"Couldn't find a place called “{wanted}”.",
                    hint="Try the city's name, or the city and country, e.g. “Pune, India”.",
                )
            self._places[key] = place
        return self._places[key]

    def _prepare_forecast(self, args: ForecastArgs) -> PreparedAction:
        wanted = (args.location or self._config.location).strip()
        if not wanted:
            raise NeedsClarificationError(
                "Which city should I get the weather for?",
                hint='Set your home city with [weather] location = "…" in the config.',
            )
        units = self._config.units

        def run(cancel: threading.Event) -> ToolResult:
            place = self._geocode(wanted)
            params: dict[str, Any] = {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            }
            if units == "fahrenheit":
                params |= {"temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}
            url = f"{FORECAST_URL}?{urllib.parse.urlencode(params)}"
            data = self._get(url, "get the weather")
            name = _place_name(place)
            try:
                message = describe_forecast(name, data, units)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    f"The weather service sent an unexpected reply: {exc!r}"
                ) from exc
            return ToolResult(
                tool="weather.forecast",
                status=ToolStatus.SUCCESS,
                message=message,
                data={"place": name, "units": units},
                untrusted=True,  # comes from an outside service
            )

        return PreparedAction(description=f"Get the weather for {wanted} from Open-Meteo", run=run)
