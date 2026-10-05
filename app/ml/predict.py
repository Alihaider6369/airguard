"""
Use the trained model to forecast PM2.5 for one place.

Kept free of any database code so it is easy to test: give it a table of
recent hourly rows, get back a forecast.

One POOLED model (trained on many Pakistani cities) is used for every place, so
any Pakistani city works, not just the ones we trained on.
"""

from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from app.ml.features import add_features, prepare_hourly

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_MODEL_NAME = "pakistan"

# Enough history for the longest lag (48h) + rolling windows, with margin.
HISTORY_HOURS_NEEDED = 96


class ModelNotFoundError(Exception):
    """No trained model file exists."""


class InsufficientDataError(Exception):
    """Not enough recent rows to build the model's input."""


@lru_cache(maxsize=4)
def load_bundle(name: str = DEFAULT_MODEL_NAME) -> dict:
    """Load (and cache) the saved model file."""
    path = MODELS_DIR / f"pm25_24h_{name.lower()}.joblib"
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found ({path.name}). Run: python -m scripts.train_model"
        )
    return joblib.load(path)


def typical_error_for(bundle: dict, city: Optional[str]) -> Optional[float]:
    """
    Average test error (µg/m³) to show next to a forecast.
    A city seen in training uses its own test error; any other city uses the
    error measured on cities the model never saw (more honest for new places).
    """
    metrics = bundle.get("metrics", {})
    by_city = metrics.get("MAE_by_city", {})
    if city and city.lower() in by_city:
        return by_city[city.lower()]
    return metrics.get("MAE_unseen") or metrics.get("MAE")


def forecast_from_dataframe(bundle: dict, raw_df: pd.DataFrame,
                            city: Optional[str] = None) -> Dict[str, object]:
    """
    Forecast PM2.5 `horizon_hours` after the newest row of `raw_df`.

    `raw_df` needs columns: timestamp, pm25, pm10, no2, co, temperature,
    humidity, wind_speed, covering at least the last ~96 hours.
    """
    if len(raw_df) < HISTORY_HOURS_NEEDED:
        raise InsufficientDataError(
            f"Need at least {HISTORY_HOURS_NEEDED} hourly rows, got {len(raw_df)}."
        )

    hourly = prepare_hourly(raw_df)
    # Fill remaining gaps with the last known value (e.g. weather that has
    # not arrived yet for the newest hours).
    features = add_features(hourly).ffill()

    row = features.iloc[[-1]][bundle["feature_columns"]]
    if row.isna().any(axis=1).iloc[0]:
        raise InsufficientDataError("Newest hour still has missing inputs.")

    prediction = float(bundle["model"].predict(row)[0])
    as_of = features.index[-1]
    horizon = timedelta(hours=bundle["horizon_hours"])

    return {
        "as_of": as_of.to_pydatetime(),
        "target_time": (as_of + horizon).to_pydatetime(),
        "current_pm25": float(row["pm25"].iloc[0]),
        "predicted_pm25": max(prediction, 0.0),  # concentration can't be negative
        "model_name": bundle.get("model_name", "unknown"),
        "typical_error": typical_error_for(bundle, city),
    }
