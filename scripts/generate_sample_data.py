"""
Generate a SYNTHETIC hourly dataset for testing the pipeline.

!! This is fake data (daily + seasonal patterns + noise). Use it only to
!! check that loading/API/dashboard work. NEVER train your final model on it
!! or present its results as real air-quality predictions.

Run from the project root:
    python -m scripts.generate_sample_data --days 180
Creates: data/sample_air_quality.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_air_quality.csv"


def generate(days: int, seed: int = 42) -> pd.DataFrame:
    """Build `days` days of hourly synthetic readings."""
    rng = np.random.default_rng(seed)  # fixed seed -> same data every time
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    hours = timestamps.hour.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    # Weather: warm in summer, cooler at night, humidity roughly opposite.
    temperature = (
        22 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        + 5 * np.sin(2 * np.pi * (hours - 9) / 24)
        + rng.normal(0, 1.5, len(timestamps))
    )
    humidity = np.clip(65 - 0.8 * (temperature - 22) + rng.normal(0, 6, len(timestamps)), 15, 100)
    wind_speed = np.clip(rng.gamma(2.0, 1.2, len(timestamps)), 0, 15)

    # PM2.5: higher in winter and at rush hours, lower when windy.
    winter_boost = 40 * np.cos(2 * np.pi * (day_of_year - 15) / 365).clip(min=0)
    rush_hour = 15 * (np.exp(-((hours - 8) ** 2) / 8) + np.exp(-((hours - 20) ** 2) / 8))
    pm25 = np.clip(
        40 + winter_boost + rush_hour - 4 * wind_speed + rng.normal(0, 8, len(timestamps)),
        3, None,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pm25": pm25,
            "pm10": pm25 * 1.5 + rng.normal(0, 10, len(timestamps)),
            "no2": 20 + 0.2 * pm25 + rng.normal(0, 4, len(timestamps)),
            "co": np.clip(0.4 + 0.01 * pm25 + rng.normal(0, 0.1, len(timestamps)), 0.05, None),
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
        }
    )
    # Round only the numeric columns (timestamps can't be rounded this way).
    return df.round({c: 2 for c in df.columns if c != "timestamp"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test data.")
    parser.add_argument("--days", type=int, default=180, help="Number of days (default 180)")
    args = parser.parse_args()

    df = generate(args.days)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
