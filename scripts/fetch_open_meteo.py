"""
Download YEARS of hourly air-quality + weather history for several Pakistani
cities and store it in the database. This is the TRAINING data for the model.
(The live website downloads the latest days by itself; see app/services.)

Run from the project root:
    python -m scripts.fetch_open_meteo                    # all cities below
    python -m scripts.fetch_open_meteo --cities Lahore Quetta
    python -m scripts.fetch_open_meteo --start 2024-01-01 --force

Cities already stored are skipped unless you pass --force.

IMPORTANT (mention this when presenting):
  * Pollution values come from the CAMS global atmospheric model (satellite +
    model estimates on a coarse ~45 km grid). They are NOT readings from
    ground sensors.
  * Weather values come from the ERA5 reanalysis (also model-based).
Open-Meteo asks users to credit CAMS (Copernicus) and Open-Meteo.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

from app import crud
from app.database import SessionLocal
from app.services.open_meteo import (
    AIR_URL,
    AIR_VARIABLES,
    WEATHER_ARCHIVE_URL,
    WEATHER_VARIABLES,
    OpenMeteoError,
    http_get_json,
    merge_air_weather,
    parse_hourly,
)

# ---- Cities used for TRAINING (name -> latitude, longitude). ---------------
# A spread across the country (north, south, plains, plateau) so the pooled
# model learns patterns that carry over to cities it has never seen.
CITIES: Dict[str, tuple] = {
    "Lahore": (31.5204, 74.3587),
    "Faisalabad": (31.4504, 73.1350),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Multan": (30.1575, 71.5249),
    "Hyderabad": (25.3960, 68.3578),
    "Gujranwala": (32.1877, 74.1945),
    "Sialkot": (32.4945, 74.5229),
    "Bahawalpur": (29.3956, 71.6836),
    "Sukkur": (27.7052, 68.8574),
    "Sargodha": (32.0836, 72.6711),
    "Abbottabad": (34.1688, 73.2215),
}

# Ask for at most this many days per request (keeps each response small).
CHUNK_DAYS = 90


def date_chunks(start: date, end: date) -> List[tuple]:
    """Split [start, end] into consecutive windows of at most CHUNK_DAYS days."""
    chunks, current = [], start
    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def _safe_get(url: str, params: dict):
    """Download one chunk; on failure print why and return None."""
    try:
        # More patient than the live API: rate limits are common in long downloads.
        return http_get_json(url, params, retries=4, backoff=15, timeout=60)
    except OpenMeteoError as error:
        print(f"  ! {error}")
        return None


def fetch_city(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Download air quality + weather for one city and merge them by hour."""
    air_parts, weather_parts = [], []

    for chunk_start, chunk_end in date_chunks(start, end):
        print(f"  {chunk_start} -> {chunk_end}")
        common = {
            "latitude": lat, "longitude": lon,
            "start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat(),
            "timezone": "UTC",  # one consistent time zone for every row
        }
        air = _safe_get(AIR_URL, {**common, "hourly": ",".join(AIR_VARIABLES)})
        weather = _safe_get(
            WEATHER_ARCHIVE_URL,
            {**common, "hourly": ",".join(WEATHER_VARIABLES), "wind_speed_unit": "ms"},
        )
        air_parts.append(parse_hourly(air, AIR_VARIABLES))
        weather_parts.append(parse_hourly(weather, WEATHER_VARIABLES))

    air_df = pd.concat([p for p in air_parts if not p.empty]) if any(not p.empty for p in air_parts) else pd.DataFrame()
    weather_df = pd.concat([p for p in weather_parts if not p.empty]) if any(not p.empty for p in weather_parts) else pd.DataFrame()
    return merge_air_weather(air_df, weather_df)


def to_rows(df: pd.DataFrame, location_id: int) -> List[dict]:
    """Convert the merged DataFrame into dicts ready for the database."""
    columns = ["pm25", "pm10", "no2", "co", "temperature", "humidity", "wind_speed"]
    rows = []
    for timestamp, record in df.iterrows():
        row = {"location_id": location_id, "timestamp": timestamp.to_pydatetime()}
        for column in columns:
            value = record.get(column)
            # NaN is not valid in SQL; store it as NULL (None) instead.
            row[column] = None if value is None or pd.isna(value) else float(value)
        rows.append(row)
    return rows


def main() -> None:
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    parser = argparse.ArgumentParser(description="Download training history into the database.")
    parser.add_argument("--cities", nargs="+", default=list(CITIES),
                        help=f"City names (default: all). Available: {', '.join(CITIES)}")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 1, 1),
                        help="Start date YYYY-MM-DD (default 2023-01-01)")
    parser.add_argument("--end", type=date.fromisoformat, default=yesterday,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true",
                        help="Download again even if the city already has data")
    args = parser.parse_args()

    lookup = {name.lower(): name for name in CITIES}
    names = []
    for requested in args.cities:
        if requested.lower() not in lookup:
            raise SystemExit(f"Unknown city '{requested}'. Available: {', '.join(CITIES)}")
        names.append(lookup[requested.lower()])

    with SessionLocal() as db:
        for name in names:
            lat, lon = CITIES[name]
            location = crud.get_or_create_location(db, name, lat, lon)

            existing = crud.count_records(db, location.id)
            if existing and not args.force:
                print(f"\n== {name}: already has {existing} records, skipping (use --force to redo) ==")
                continue

            print(f"\n== {name} ({lat}, {lon}) ==")
            df = fetch_city(lat, lon, args.start, args.end)
            if df.empty:
                print(f"No pollution data received for {name}. Nothing stored.")
                continue

            print(f"Received {len(df)} hourly rows, {df.index.min()} -> {df.index.max()}, "
                  f"PM2.5 present in {df['pm25'].notna().sum()} rows")
            crud.bulk_insert_records(db, to_rows(df, location.id))
            print(f"'{name}' now has {crud.count_records(db, location.id)} records in the database.")


if __name__ == "__main__":
    main()
