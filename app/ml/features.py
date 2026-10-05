"""
Feature engineering for the 24-hour PM2.5 forecast.

The goal: standing at hour t, predict PM2.5 at hour t+24.

LEAKAGE RULE: every feature below uses ONLY information from hour t or
earlier. The target (PM2.5 at t+24) is never visible to the model as input.
This file is shared by training (scripts/train_model.py) and, later, by the
FastAPI /predict endpoint, so both build features in exactly the same way.
"""

import numpy as np
import pandas as pd

# How far ahead we predict.
HORIZON_HOURS = 24

# Columns that come straight from the database.
BASE_COLUMNS = ["pm25", "pm10", "no2", "co", "temperature", "humidity", "wind_speed"]

# "PM2.5 N hours ago" features. 24/48 capture the daily rhythm.
LAG_HOURS = [1, 3, 6, 12, 24, 48]


def prepare_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put raw database rows into a clean, strictly hourly table.

    - sorts by time and removes duplicate hours
    - inserts any missing hours as empty rows (so "24 rows back" always means
      "24 hours back")
    - fills tiny gaps (up to 3 hours) by interpolation
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp").asfreq("h")
    df[BASE_COLUMNS] = df[BASE_COLUMNS].astype(float).interpolate(limit=3)
    return df[BASE_COLUMNS]


def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build the model's input columns from a table made by prepare_hourly()."""
    pm25 = hourly["pm25"]
    out = hourly[BASE_COLUMNS].copy()  # current conditions (hour t)

    # Recent history of PM2.5.
    for lag in LAG_HOURS:
        out[f"pm25_lag_{lag}h"] = pm25.shift(lag)

    # Short and daily trends (windows end at hour t, so no future info).
    out["pm25_mean_6h"] = pm25.rolling(6).mean()
    out["pm25_mean_24h"] = pm25.rolling(24).mean()
    out["pm25_std_24h"] = pm25.rolling(24).std()
    out["pm25_change_3h"] = pm25 - pm25.shift(3)

    # Calendar features. Hour is cyclic (23h is next to 0h), so use sin/cos.
    hour = hourly.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = hourly.index.dayofweek
    out["month"] = hourly.index.month
    return out


def add_target(hourly: pd.DataFrame) -> pd.Series:
    """The answer we want to predict: PM2.5 exactly HORIZON_HOURS later."""
    return hourly["pm25"].shift(-HORIZON_HOURS).rename("target_pm25")
