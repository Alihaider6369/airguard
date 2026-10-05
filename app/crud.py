"""
CRUD helpers: all reusable database operations live here.

Keeping queries in one file means the API routes, the ML training script
and the data loader all share the same tested functions instead of each
writing their own SQL.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AirQualityRecord, Location, Prediction

# How many rows to insert per database call when loading big CSV files.
INSERT_CHUNK_SIZE = 1000


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
def get_location_by_name(db: Session, name: str) -> Optional[Location]:
    """Return the location with this name (any letter case), or None."""
    return db.scalar(
        select(Location).where(func.lower(Location.name) == name.strip().lower())
    )


def get_or_create_location(
    db: Session,
    name: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> Location:
    """Fetch a location by name, creating it first if it is missing."""
    location = get_location_by_name(db, name)
    if location is None:
        location = Location(name=name, latitude=latitude, longitude=longitude)
        db.add(location)
        db.commit()
        db.refresh(location)  # loads the auto-generated id
    return location


def list_locations(db: Session) -> List[Location]:
    """Return all locations, sorted by name."""
    return list(db.scalars(select(Location).order_by(Location.name)))


# --------------------------------------------------------------------------
# Air-quality records
# --------------------------------------------------------------------------
def bulk_insert_records(db: Session, rows: List[Dict]) -> int:
    """
    Insert many hourly records quickly, skipping duplicates.

    Each row is a dict with keys matching AirQualityRecord columns
    (must include location_id and timestamp). A row whose
    (location_id, timestamp) already exists is silently ignored, so running
    the loader twice will not create duplicate data.

    Returns the number of rows that were sent to the database.
    """
    if not rows:
        return 0

    # "ON CONFLICT DO NOTHING" is dialect-specific, so pick the right insert.
    dialect = db.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise NotImplementedError(f"Bulk insert not supported for {dialect}")

    statement = insert(AirQualityRecord).on_conflict_do_nothing(
        index_elements=["location_id", "timestamp"]
    )

    # Insert in chunks so very large files don't use too much memory.
    for start in range(0, len(rows), INSERT_CHUNK_SIZE):
        db.execute(statement, rows[start : start + INSERT_CHUNK_SIZE])
    db.commit()
    return len(rows)


def get_records(
    db: Session,
    location_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[AirQualityRecord]:
    """Return a location's records in time order, optionally within a range."""
    query = select(AirQualityRecord).where(
        AirQualityRecord.location_id == location_id
    )
    if start is not None:
        query = query.where(AirQualityRecord.timestamp >= start)
    if end is not None:
        query = query.where(AirQualityRecord.timestamp <= end)
    return list(db.scalars(query.order_by(AirQualityRecord.timestamp)))


def get_latest_records(
    db: Session, location_id: int, limit: int = 24
) -> List[AirQualityRecord]:
    """
    Return the newest `limit` records, oldest first.

    Useful for building model input (lags / rolling averages) and for the
    "current conditions" card on the dashboard.
    """
    newest_first = db.scalars(
        select(AirQualityRecord)
        .where(AirQualityRecord.location_id == location_id)
        .order_by(AirQualityRecord.timestamp.desc())
        .limit(limit)
    )
    return list(reversed(list(newest_first)))


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def save_prediction(
    db: Session,
    location_id: int,
    target_time: datetime,
    predicted_pm25: float,
    model_name: Optional[str] = None,
) -> Prediction:
    """Store one forecast made by the ML model."""
    prediction = Prediction(
        location_id=location_id,
        target_time=target_time,
        predicted_pm25=predicted_pm25,
        model_name=model_name,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_latest_prediction(db: Session, location_id: int) -> Optional[Prediction]:
    """Return the most recently made prediction for a location, if any."""
    return db.scalar(
        select(Prediction)
        .where(Prediction.location_id == location_id)
        .order_by(Prediction.made_at.desc())
        .limit(1)
    )


def get_prediction(
    db: Session, location_id: int, target_time: datetime, model_name: str
) -> Optional[Prediction]:
    """Find an existing prediction for the same city, target hour and model."""
    return db.scalar(
        select(Prediction).where(
            Prediction.location_id == location_id,
            Prediction.target_time == target_time,
            Prediction.model_name == model_name,
        )
    )


def count_records(db: Session, location_id: int) -> int:
    """How many hourly records are stored for a location."""
    return db.scalar(
        select(func.count(AirQualityRecord.id)).where(
            AirQualityRecord.location_id == location_id
        )
    ) or 0
