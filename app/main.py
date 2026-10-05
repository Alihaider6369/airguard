"""
AirGuard API (FastAPI).

Works for ANY city/town in Pakistan: the name is converted to coordinates,
the latest data is downloaded live, and a pooled model forecasts PM2.5 for the
next 24 hours.

Run from the project root:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs and try /api/dashboard?city=Multan
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import crud, models  # noqa: F401  (models import registers the tables)
from app.aqi import describe_pm25
from app.config import settings
from app.database import Base, engine, get_db
from app.ml.predict import (
    InsufficientDataError,
    ModelNotFoundError,
    forecast_from_dataframe,
    load_bundle,
)
from app.schemas import (
    CurrentOut,
    DashboardOut,
    ForecastOut,
    HistoryOut,
    PlaceOut,
    ReadingOut,
    StatusOut,
)
from app.services.geocoding import (
    OutsidePakistanError, Place, PlaceNotFoundError, place_from_coordinates,
    resolve_city, search_places,
)
from app.services.live_data import get_recent_data
from app.services.open_meteo import OpenMeteoError


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Runs once when the server starts: make sure the tables exist."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="AirGuard API",
    version="0.2.0",
    description=(
        "Live air quality and 24-hour PM2.5 forecasts for any city in Pakistan. "
        "Pollution data are CAMS model estimates (via Open-Meteo), not ground-sensor readings."
    ),
    lifespan=lifespan,
)

# Allow the website (frontend) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

DbSession = Annotated[Session, Depends(get_db)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _place_out(place: Place) -> PlaceOut:
    return PlaceOut(
        name=place.name, region=place.region, display_name=place.display_name,
        latitude=place.latitude, longitude=place.longitude,
    )


def _clean(value):
    """NaN -> None so it becomes JSON null."""
    return None if value is None or pd.isna(value) else round(float(value), 2)


def _reading(timestamp: pd.Timestamp, row: pd.Series) -> ReadingOut:
    return ReadingOut(
        timestamp=timestamp.to_pydatetime(),
        pm25=_clean(row.get("pm25")), pm10=_clean(row.get("pm10")),
        no2=_clean(row.get("no2")), co=_clean(row.get("co")),
        temperature=_clean(row.get("temperature")), humidity=_clean(row.get("humidity")),
        wind_speed=_clean(row.get("wind_speed")),
    )


def _age_hours(timestamp: datetime) -> float:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return round((now - timestamp).total_seconds() / 3600, 1)


def _load(city: Optional[str], lat: Optional[float] = None, lon: Optional[float] = None):
    """city name OR coordinates -> (place, recent hourly DataFrame), with friendly errors."""
    try:
        if lat is not None and lon is not None:
            place = place_from_coordinates(lat, lon)
        elif city:
            place = resolve_city(city)
        else:
            raise HTTPException(422, "Give a city name, or lat and lon.")
        return place, get_recent_data(place)
    except PlaceNotFoundError:
        raise HTTPException(404, f"'{city}' was not found in Pakistan. Check the spelling.")
    except OutsidePakistanError:
        raise HTTPException(422, "This location is outside Pakistan. AirGuard covers Pakistan only.")
    except OpenMeteoError as error:
        raise HTTPException(502, f"Data provider problem: {error}")


def _current(place: Place, df: pd.DataFrame) -> CurrentOut:
    last = df["pm25"].last_valid_index()
    if last is None:
        raise HTTPException(404, f"No PM2.5 data available for {place.display_name}.")
    reading = _reading(last, df.loc[last])
    return CurrentOut(
        place=_place_out(place), reading=reading,
        status=StatusOut(**describe_pm25(reading.pm25)),
        data_age_hours=_age_hours(reading.timestamp),
    )


def _forecast(place: Place, df: pd.DataFrame, db: Session, log: bool = True) -> ForecastOut:
    raw_df = df.reset_index()  # index is named "timestamp" -> becomes a column
    try:
        result = forecast_from_dataframe(load_bundle(), raw_df, city=place.name)
    except ModelNotFoundError as error:
        raise HTTPException(503, str(error))
    except InsufficientDataError as error:
        raise HTTPException(422, str(error))

    # Remember the forecast (best effort: never break the response over logging).
    try:
        if not log:  # GPS lookups are never stored (privacy)
            raise RuntimeError
        location = crud.get_or_create_location(db, place.name, place.latitude, place.longitude)
        if crud.get_prediction(db, location.id, result["target_time"], result["model_name"]) is None:
            crud.save_prediction(db, location.id, result["target_time"],
                                 result["predicted_pm25"], result["model_name"])
    except RuntimeError:
        pass
    except Exception:
        db.rollback()

    # Rising / falling / stable: compare forecast with now (10% dead-zone).
    change = result["predicted_pm25"] - result["current_pm25"]
    threshold = 0.10 * max(result["current_pm25"], 1.0)
    trend = "rising" if change > threshold else "falling" if change < -threshold else "stable"

    return ForecastOut(
        place=_place_out(place),
        as_of=result["as_of"], target_time=result["target_time"],
        current_pm25=round(result["current_pm25"], 1),
        predicted_pm25=round(result["predicted_pm25"], 1),
        typical_error=None if result["typical_error"] is None else round(result["typical_error"], 1),
        trend=trend,
        status=StatusOut(**describe_pm25(result["predicted_pm25"])),
        model_name=result["model_name"],
        note="Based on model-estimated (CAMS) data, not ground sensors. "
             "Typical error is measured on past unseen data.",
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
PAGE = Path(__file__).resolve().parent.parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def root():
    """The dashboard page (falls back to a small JSON message)."""
    return FileResponse(PAGE) if PAGE.exists() else {"name": "AirGuard API", "docs": "/docs"}


@app.get("/api/health", tags=["system"])
def health():
    """Simple check that the server is running."""
    return {"status": "ok"}


@app.get("/api/places/search", response_model=List[PlaceOut], tags=["places"])
def places_search(q: str = Query(..., min_length=2, examples=["Mult"])):
    """Find Pakistani cities/towns by name (for the search box)."""
    try:
        return [_place_out(p) for p in search_places(q)]
    except OpenMeteoError as error:
        raise HTTPException(502, f"Data provider problem: {error}")


@app.get("/api/places/reverse", response_model=PlaceOut, tags=["places"])
def places_reverse(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    """Which Pakistani place are these GPS coordinates in?"""
    try:
        return _place_out(place_from_coordinates(lat, lon))
    except OutsidePakistanError:
        raise HTTPException(422, "This location is outside Pakistan. AirGuard covers Pakistan only.")


@app.get("/api/air-quality/current", response_model=CurrentOut, tags=["air quality"])
def current_air_quality(city: Optional[str] = Query(None, examples=["Lahore"]), lat: Optional[float] = Query(None, ge=-90, le=90), lon: Optional[float] = Query(None, ge=-180, le=180)):
    """The newest reading for any Pakistani city, with an AQI-style status."""
    place, df = _load(city, lat, lon)
    return _current(place, df)


@app.get("/api/air-quality/history", response_model=HistoryOut, tags=["air quality"])
def air_quality_history(
    city: Optional[str] = Query(None, examples=["Lahore"]), lat: Optional[float] = Query(None, ge=-90, le=90), lon: Optional[float] = Query(None, ge=-180, le=180),
    hours: int = Query(168, ge=1, le=168, description="Recent hours to return (max 7 days)"),
):
    """Recent hourly readings, oldest first (for charts)."""
    place, df = _load(city, lat, lon)
    recent = df.tail(hours)
    return HistoryOut(
        place=_place_out(place), hours=len(recent),
        readings=[_reading(ts, row) for ts, row in recent.iterrows()],
    )


@app.get("/api/forecast", response_model=ForecastOut, tags=["forecast"])
def forecast(db: DbSession, city: Optional[str] = Query(None, examples=["Lahore"]), lat: Optional[float] = Query(None, ge=-90, le=90), lon: Optional[float] = Query(None, ge=-180, le=180)):
    """24-hour-ahead PM2.5 forecast for any Pakistani city."""
    place, df = _load(city, lat, lon)
    return _forecast(place, df, db, log=lat is None)


@app.get("/api/dashboard", response_model=DashboardOut, tags=["dashboard"])
def dashboard(db: DbSession, city: Optional[str] = Query(None, examples=["Lahore"]), lat: Optional[float] = Query(None, ge=-90, le=90), lon: Optional[float] = Query(None, ge=-180, le=180)):
    """Current status + 7-day history + forecast in ONE request (what the page uses)."""
    place, df = _load(city, lat, lon)
    return DashboardOut(
        place=_place_out(place),
        current=_current(place, df),
        history=[_reading(ts, row) for ts, row in df.iterrows()],
        forecast=_forecast(place, df, db, log=lat is None),
    )
