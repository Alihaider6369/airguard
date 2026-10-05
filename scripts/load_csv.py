"""
Load a CSV file of hourly readings into the database.

Run from the project root, for example:
    python -m scripts.load_csv --csv data/sample_air_quality.csv \
        --location "Lahore" --lat 31.52 --lon 74.36

Your real dataset will have different column names. Edit COLUMN_MAP below
(left side = column name in YOUR csv, right side = database column).
Columns missing from your CSV are simply stored as empty (NULL).
"""

import argparse
from pathlib import Path

import pandas as pd

from app import crud
from app.database import SessionLocal

# ---- EDIT THIS to match your dataset's column names -----------------------
COLUMN_MAP = {
    "timestamp": "timestamp",      # e.g. "date", "datetime", "Date Time"
    "pm25": "pm25",                # e.g. "PM2.5", "pm2_5"
    "pm10": "pm10",
    "no2": "no2",
    "co": "co",
    "temperature": "temperature",  # e.g. "temp", "temperature_2m"
    "humidity": "humidity",        # e.g. "relative_humidity_2m"
    "wind_speed": "wind_speed",    # e.g. "windspeed_10m"
}
# ---------------------------------------------------------------------------

# Database columns we store (everything except id / location_id).
VALUE_COLUMNS = ["pm25", "pm10", "no2", "co", "temperature", "humidity", "wind_speed"]


def read_and_clean(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and return a clean DataFrame with database column names."""
    df = pd.read_csv(csv_path)

    # Rename the CSV's columns to our database column names.
    df = df.rename(columns=COLUMN_MAP)
    if "timestamp" not in df.columns:
        raise SystemExit(
            "No timestamp column found. Edit COLUMN_MAP at the top of this file "
            f"(your CSV columns are: {list(df.columns)})"
        )

    # Parse timestamps; unreadable ones become NaT and are dropped.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # If the CSV has timezone info, convert to UTC and drop the tz marker,
    # so every row is stored in the same consistent format.
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Add any missing value columns as empty, and force numeric types
    # (text like "N/A" becomes NaN instead of crashing the load).
    for column in VALUE_COLUMNS:
        if column not in df.columns:
            df[column] = None
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep one row per timestamp and sort oldest -> newest.
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df[["timestamp"] + VALUE_COLUMNS]


def to_rows(df: pd.DataFrame, location_id: int) -> list:
    """Convert the DataFrame to a list of dicts ready for the database."""
    rows = []
    for record in df.to_dict("records"):
        row = {"location_id": location_id, "timestamp": record["timestamp"].to_pydatetime()}
        for column in VALUE_COLUMNS:
            value = record[column]
            # NaN is not valid in SQL; store it as NULL (None) instead.
            row[column] = None if pd.isna(value) else float(value)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a CSV into the database.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the CSV file")
    parser.add_argument("--location", required=True, help="City / station name")
    parser.add_argument("--lat", type=float, default=None, help="Latitude (optional)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude (optional)")
    args = parser.parse_args()

    df = read_and_clean(args.csv)
    print(f"Read {len(df)} valid rows from {args.csv}")

    with SessionLocal() as db:
        location = crud.get_or_create_location(db, args.location, args.lat, args.lon)
        crud.bulk_insert_records(db, to_rows(df, location.id))

        # Report what is actually in the database now (duplicates were skipped).
        stored = len(crud.get_records(db, location.id))
        print(f"Location '{location.name}' now has {stored} records in the database.")


if __name__ == "__main__":
    main()
