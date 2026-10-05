"""
Latest hourly data for any place, cached for a while.

The cache protects the free Open-Meteo API (and makes the dashboard fast):
the same city asked again within CACHE_SECONDS is served from memory.
"""

import time
from typing import Dict, Tuple

import pandas as pd

from app.services.geocoding import Place
from app.services.open_meteo import fetch_recent

CACHE_SECONDS = 30 * 60
_cache: Dict[Tuple[float, float], Tuple[float, pd.DataFrame]] = {}


def get_recent_data(place: Place) -> pd.DataFrame:
    """Last ~7 days of hourly data for `place` (downloaded at most every 30 min)."""
    key = (round(place.latitude, 3), round(place.longitude, 3))
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]

    df = fetch_recent(place.latitude, place.longitude)
    _cache[key] = (time.monotonic(), df)
    return df
