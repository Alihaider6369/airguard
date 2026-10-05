"""
Shared helpers for talking to the free Open-Meteo APIs.

Used by BOTH the data-download script (history, for training) and the live API
(last few days, for forecasting), so the two always produce the same columns
and units.

Data caveat: pollution = CAMS atmospheric model estimates, weather = model data
(ERA5 for history, Open-Meteo's forecast models for the live days). Not sensors.
Open-Meteo's free API is for non-commercial use; please credit Open-Meteo / CAMS.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo variable name -> our database / feature column name.
AIR_VARIABLES = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "carbon_monoxide": "co",
}
WEATHER_VARIABLES = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
}

# How many recent days the live API downloads (needs >= 4 days for the model).
LIVE_PAST_DAYS = 7


class OpenMeteoError(Exception):
    """Open-Meteo could not be reached or returned an error."""


def http_get_json(url: str, params: dict, retries: int = 2,
                  backoff: float = 1.5, timeout: int = 30) -> dict:
    """
    GET `url` with query `params`; return parsed JSON.
    Retries on rate-limit (429) and server (5xx) errors, then raises OpenMeteoError.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    last_problem = "unknown error"
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(full_url, headers={"User-Agent": "AirGuard-student-project"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:200]
            last_problem = f"HTTP {error.code}: {detail}"
            if error.code != 429 and error.code < 500:
                break  # a client error will not fix itself by retrying
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_problem = f"network problem: {error}"
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    raise OpenMeteoError(f"{url} -> {last_problem}")


def parse_hourly(payload: Optional[dict], rename: Dict[str, str]) -> pd.DataFrame:
    """Open-Meteo JSON -> DataFrame indexed by hour, with our column names."""
    if not payload or "hourly" not in payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload["hourly"])
    df["timestamp"] = pd.to_datetime(df.pop("time"))
    return df.rename(columns=rename).set_index("timestamp")


def merge_air_weather(air_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Join pollution and weather by hour, fix CO units, drop empty hours."""
    if air_df.empty:
        return pd.DataFrame()
    air_df = air_df.copy()
    # Open-Meteo reports carbon monoxide in ug/m3; our table stores mg/m3.
    if "co" in air_df.columns:
        air_df["co"] = air_df["co"] / 1000.0

    merged = air_df.join(weather_df, how="left") if not weather_df.empty else air_df
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()

    pollutants = [c for c in ("pm25", "pm10", "no2", "co") if c in merged.columns]
    return merged.dropna(subset=pollutants, how="all")


def fetch_recent(latitude: float, longitude: float, past_days: int = LIVE_PAST_DAYS) -> pd.DataFrame:
    """
    Download the last `past_days` days of hourly data up to the current hour.
    Returns a DataFrame indexed by UTC hour with the model's input columns.
    """
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": 1,
    }
    air = http_get_json(AIR_URL, {**common, "hourly": ",".join(AIR_VARIABLES)})
    weather = http_get_json(
        WEATHER_FORECAST_URL,
        {**common, "hourly": ",".join(WEATHER_VARIABLES), "wind_speed_unit": "ms"},
    )
    merged = merge_air_weather(
        parse_hourly(air, AIR_VARIABLES), parse_hourly(weather, WEATHER_VARIABLES)
    )
    if merged.empty:
        raise OpenMeteoError("No pollution data returned for this location.")

    # The response also contains future hours (model forecasts): keep only the past.
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).floor("h")
    return merged[merged.index <= now]
