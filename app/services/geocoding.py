"""
Turn a city name typed by the user into coordinates (Pakistan only).
Uses the free Open-Meteo Geocoding API, so ANY Pakistani city/town it knows works.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

from app.services.open_meteo import OpenMeteoError, http_get_json

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"


class PlaceNotFoundError(Exception):
    """No place with this name was found in Pakistan."""


class OutsidePakistanError(Exception):
    """The coordinates are not inside Pakistan."""


# Rough bounding box of Pakistan: lat_min, lat_max, lon_min, lon_max.
PAKISTAN_BOUNDS = (23.5, 37.2, 60.8, 77.9)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_reverse_cache: dict = {}


@dataclass(frozen=True)
class Place:
    name: str
    region: Optional[str]   # province / admin area, e.g. "Punjab"
    latitude: float
    longitude: float

    @property
    def display_name(self) -> str:
        return f"{self.name}, {self.region}" if self.region else self.name


def parse_places(payload: dict) -> List[Place]:
    """Keep only results in Pakistan and remove near-duplicates."""
    places, seen = [], set()
    for item in payload.get("results") or []:
        if item.get("country_code") != "PK":
            continue
        key = (item["name"].lower(), round(item["latitude"], 1), round(item["longitude"], 1))
        if key in seen:
            continue
        seen.add(key)
        places.append(Place(
            name=item["name"],
            region=item.get("admin1"),
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
        ))
    return places


def search_places(query: str, limit: int = 5) -> List[Place]:
    """Return up to `limit` Pakistani places matching `query` (best match first)."""
    query = query.strip()
    if len(query) < 2:
        return []
    payload = http_get_json(GEOCODING_URL, {
        "name": query,
        "count": 20,            # ask for extra; non-Pakistan results get filtered out
        "language": "en",
        "format": "json",
        "countryCode": "PK",
    })
    return parse_places(payload)[:limit]


@lru_cache(maxsize=512)
def _resolve_cached(query_lower: str) -> Place:
    places = search_places(query_lower, limit=1)
    if not places:
        raise PlaceNotFoundError(query_lower)
    return places[0]


def resolve_city(query: str) -> Place:
    """The single best Pakistani match for a city name (cached)."""
    return _resolve_cached(query.strip().lower())


def place_from_coordinates(latitude: float, longitude: float) -> Place:
    """
    Turn the user's GPS position into a Place (e.g. "Multan, Punjab").
    Coordinates are rounded to 2 decimals (~1 km): enough for a 45 km data grid,
    and the exact position is never stored.
    """
    lat_min, lat_max, lon_min, lon_max = PAKISTAN_BOUNDS
    if not (lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max):
        raise OutsidePakistanError()
    lat, lon = round(latitude, 2), round(longitude, 2)

    if (lat, lon) not in _reverse_cache:
        try:  # the name lookup is nice-to-have: failures fall back to "Your location"
            address = http_get_json(NOMINATIM_URL, {
                "format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10,
                "addressdetails": 1, "accept-language": "en"}).get("address", {})
        except OpenMeteoError:
            return Place("Your location", None, lat, lon)
        if address.get("country_code", "pk") != "pk":
            raise OutsidePakistanError()
        name = (address.get("city") or address.get("town") or address.get("village")
                or address.get("county") or address.get("state_district"))
        _reverse_cache[(lat, lon)] = (name, address.get("state"))
    name, region = _reverse_cache[(lat, lon)]
    return Place(name or "Your location", region, lat, lon)


__all__ = ["place_from_coordinates", "OutsidePakistanError", "Place", "PlaceNotFoundError", "OpenMeteoError", "resolve_city", "search_places", "parse_places"]
