"""
Train ONE pooled model that forecasts PM2.5 24 hours ahead for ANY Pakistani city.

Run from the project root (after scripts.fetch_open_meteo):
    python -m scripts.train_model
    python -m scripts.train_model --holdout Quetta Sukkur

Why one pooled model? It learns from many cities at once (their pollution
history + weather), so it can forecast a city it has never seen, which is what
lets the website accept any city name in Pakistan.

Three candidate models are compared (Ridge, Random Forest, Gradient Boosting)
and the best one is saved.

How we keep the evaluation honest:
  * TIME-BASED split: train on the older 80% of the timeline, test on the
    newest 20% (a random split would let the model "peek" at nearby hours).
    A 24-hour gap separates them because targets reach 24 h into the future.
  * A BASELINE ("tomorrow will be the same as now") that a useful model must beat.
  * UNSEEN-CITY TEST: the best model is retrained WITHOUT some cities
    (default Quetta, Sukkur) and tested on exactly those cities. This is the
    honest evidence that the model works for places it was not trained on.
  * Note: picking the best of three models by test error is a very small
    optimistic bias. We mention it rather than hide it.

Outputs:
    models/pm25_24h_pakistan.joblib     the model + its metrics (used by the API)
    reports/forecast_<city>.png         forecast vs actual (Lahore, Faisalabad)
    reports/feature_importance.png
"""

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files only; no pop-up window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import crud
from app.database import SessionLocal
from app.ml.features import (
    BASE_COLUMNS,
    HORIZON_HOURS,
    add_features,
    add_target,
    prepare_hourly,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_FILE_NAME = "pakistan"
DEFAULT_HOLDOUT = ["Quetta", "Sukkur"]
PLOT_CITIES = ["Lahore", "Faisalabad"]


def make_candidates() -> Dict[str, object]:
    """The models we compare. Sizes are limited so the saved file stays light."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=40, max_features=0.5,
            n_jobs=-1, random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
    }


def load_city_dataframe(db, location_id: int) -> pd.DataFrame:
    """Read one city's records from the database into a DataFrame."""
    records = crud.get_records(db, location_id)
    return pd.DataFrame(
        [{"timestamp": r.timestamp, **{c: getattr(r, c) for c in BASE_COLUMNS}} for r in records]
    )


def score(y_true, y_pred) -> Dict[str, float]:
    """MAE / RMSE are in PM2.5 units (µg/m³); R² is 1.0 for a perfect model."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_city_frame(raw_df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Features + target for one city (rows with any missing value are dropped)."""
    hourly = prepare_hourly(raw_df)
    frame = add_features(hourly).join(add_target(hourly)).dropna()
    frame["city"] = city.lower()
    return frame


def split_by_time(frames: Dict[str, pd.DataFrame], test_fraction: float):
    """One shared cut-off date for every city (no city sees the other's future)."""
    all_times = None
    for frame in frames.values():
        all_times = frame.index if all_times is None else all_times.union(frame.index)
    cutoff = all_times[int(len(all_times) * (1 - test_fraction))]
    gap = pd.Timedelta(hours=HORIZON_HOURS)

    train = pd.concat([f[f.index < cutoff - gap] for f in frames.values()])
    test = pd.concat([f[f.index >= cutoff] for f in frames.values()])
    return train, test


def train_and_evaluate(frames: Dict[str, pd.DataFrame], holdout: List[str],
                       test_fraction: float = 0.2) -> dict:
    """Train all candidates on the pooled data, keep the best, test on unseen cities."""
    train, test = split_by_time(frames, test_fraction)
    feature_columns = [c for c in train.columns if c not in ("target_pm25", "city")]
    x_train, y_train = train[feature_columns], train["target_pm25"]
    x_test, y_test = test[feature_columns], test["target_pm25"]

    # 1) Train every candidate on all cities; score on the newest, unseen period.
    fitted, metrics, predictions = {}, {}, {}
    for name, model in make_candidates().items():
        print(f"  training {name} ...")
        model.fit(x_train, y_train)
        fitted[name] = model
        predictions[name] = model.predict(x_test)
        metrics[name] = score(y_test, predictions[name])

    best_name = min(metrics, key=lambda n: metrics[n]["MAE"])
    best_model = fitted[best_name]
    test = test.assign(prediction=predictions[best_name])
    baseline_metrics = score(y_test, test["pm25"])

    # 2) Per-city accuracy of the saved model on the test period.
    by_city = {}
    for city, group in test.groupby("city"):
        by_city[city] = {
            "model": score(group["target_pm25"], group["prediction"]),
            "baseline": score(group["target_pm25"], group["pm25"]),
        }

    # 3) UNSEEN-CITY TEST: retrain the winning model type without the held-out cities.
    unseen = None
    held = [c.lower() for c in holdout if c.lower() in frames]
    if held and len(frames) - len(held) >= 2:
        print(f"  unseen-city check: retraining {best_name} without {', '.join(held)} ...")
        mask_train = ~train["city"].isin(held)
        check_model = clone(make_candidates()[best_name]).fit(
            x_train[mask_train], y_train[mask_train])
        held_test = test[test["city"].isin(held)]
        pred_held = check_model.predict(held_test[feature_columns])
        unseen = {
            "cities": held,
            "model": score(held_test["target_pm25"], pred_held),
            "baseline": score(held_test["target_pm25"], held_test["pm25"]),
        }

    # 4) Which inputs mattered (works for any model type).
    sample = test.sample(n=min(3000, len(test)), random_state=42)
    result = permutation_importance(
        best_model, sample[feature_columns], sample["target_pm25"],
        scoring="neg_mean_absolute_error", n_repeats=3, random_state=42)

    return {
        "model": best_model, "model_name": best_name,
        "feature_columns": feature_columns,
        "train_rows": len(train), "test_rows": len(test),
        "train_period": (train.index.min(), train.index.max()),
        "test_period": (test.index.min(), test.index.max()),
        "metrics_all": metrics, "metrics_best": metrics[best_name],
        "metrics_baseline": baseline_metrics,
        "by_city": by_city, "unseen": unseen, "test": test,
        "importance": pd.Series(result.importances_mean, index=feature_columns),
    }


def print_report(result: dict) -> None:
    """Print the key numbers in small tables."""
    b = result["metrics_baseline"]
    print(f"\nTrain: {result['train_rows']} rows "
          f"({result['train_period'][0]:%Y-%m-%d} -> {result['train_period'][1]:%Y-%m-%d})")
    print(f"Test : {result['test_rows']} rows "
          f"({result['test_period'][0]:%Y-%m-%d} -> {result['test_period'][1]:%Y-%m-%d})")

    print(f"\n=== All cities together (unseen period) ===")
    print(f"{'':26}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    print(f"{'Baseline (same as now)':26}{b['MAE']:8.2f}{b['RMSE']:8.2f}{b['R2']:8.3f}")
    for name, m in result["metrics_all"].items():
        marker = "  <- saved" if name == result["model_name"] else ""
        print(f"{name:26}{m['MAE']:8.2f}{m['RMSE']:8.2f}{m['R2']:8.3f}{marker}")

    print(f"\n=== Per city, saved model: {result['model_name']} (MAE in µg/m³) ===")
    print(f"{'city':14}{'baseline':>10}{'model':>8}{'better by':>11}")
    for city, m in sorted(result["by_city"].items()):
        gain = (1 - m["model"]["MAE"] / m["baseline"]["MAE"]) * 100
        print(f"{city:14}{m['baseline']['MAE']:10.2f}{m['model']['MAE']:8.2f}{gain:10.1f}%")

    u = result["unseen"]
    if u:
        gain = (1 - u["model"]["MAE"] / u["baseline"]["MAE"]) * 100
        print(f"\n=== UNSEEN cities ({', '.join(u['cities'])}): model never saw them in training ===")
        print(f"baseline MAE {u['baseline']['MAE']:.2f}  |  model MAE {u['model']['MAE']:.2f}  "
              f"|  better by {gain:.1f}%  |  R2 {u['model']['R2']:.3f}")


def save_plots(result: dict, reports_dir: Path, days: int = 14) -> None:
    """Forecast-vs-actual graphs for the plot cities + a feature-importance graph."""
    for city in PLOT_CITIES:
        group = result["test"][result["test"]["city"] == city.lower()].tail(days * 24)
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(group.index, group["target_pm25"], label="Actual PM2.5", color="black", linewidth=1.6)
        ax.plot(group.index, group["prediction"], color="tab:red", linewidth=1.3,
                label=f"Forecast made 24h earlier ({result['model_name']})")
        ax.plot(group.index, group["pm25"], label="Baseline (same as now)",
                color="tab:blue", alpha=0.5, linewidth=1)
        ax.set_title(f"{city}: PM2.5 forecast vs actual (last {days} days of test data)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(reports_dir / f"forecast_{city.lower()}.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    result["importance"].nlargest(10).sort_values().plot.barh(ax=ax, color="tab:green")
    ax.set_title("Top 10 most important features (permutation importance)")
    ax.set_xlabel("Increase in error (µg/m³) when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(reports_dir / "feature_importance.png", dpi=130)
    plt.close(fig)


def run_training(city_frames: Dict[str, pd.DataFrame], holdout: List[str],
                 models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    """Train, report, and save everything."""
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    result = train_and_evaluate(city_frames, holdout)
    print_report(result)

    metrics = {
        **result["metrics_best"],
        "baseline_MAE": result["metrics_baseline"]["MAE"],
        "MAE_by_city": {c: m["model"]["MAE"] for c, m in result["by_city"].items()},
    }
    if result["unseen"]:
        metrics["MAE_unseen"] = result["unseen"]["model"]["MAE"]
        metrics["baseline_MAE_unseen"] = result["unseen"]["baseline"]["MAE"]
        metrics["unseen_cities"] = result["unseen"]["cities"]

    path = models_dir / f"pm25_24h_{MODEL_FILE_NAME}.joblib"
    joblib.dump(
        {
            "model": result["model"],
            "model_name": result["model_name"],
            "feature_columns": result["feature_columns"],
            "horizon_hours": HORIZON_HOURS,
            "trained_cities": sorted(city_frames),
            "metrics": metrics,
            "test_period": [str(t) for t in result["test_period"]],
        },
        path,
        compress=3,  # smaller file (GitHub rejects files over 100 MB)
    )
    save_plots(result, reports_dir)
    print(f"\nSaved model  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved graphs -> {reports_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pooled 24h PM2.5 forecast model.")
    parser.add_argument("--holdout", nargs="*", default=DEFAULT_HOLDOUT,
                        help="Cities kept out of the unseen-city check's training "
                             f"(default: {' '.join(DEFAULT_HOLDOUT)})")
    args = parser.parse_args()

    with SessionLocal() as db:
        locations = crud.list_locations(db)
        if len(locations) < 4:
            raise SystemExit(
                f"Only {len(locations)} cities in the database. Run "
                "'python -m scripts.fetch_open_meteo' first (it loads 15 cities)."
            )
        frames = {}
        for location in locations:
            raw_df = load_city_dataframe(db, location.id)
            if len(raw_df) < 24 * 60:  # need at least ~2 months of data
                print(f"Skipping {location.name}: only {len(raw_df)} rows.")
                continue
            frames[location.name.lower()] = build_city_frame(raw_df, location.name)

    print(f"Training on {len(frames)} cities: {', '.join(sorted(frames))}")
    run_training(frames, args.holdout)


if __name__ == "__main__":
    main()
