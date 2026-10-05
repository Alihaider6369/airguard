"""
Database tables (ORM models).

Three tables, kept deliberately simple:

    locations            -> a city / monitoring station
    air_quality_records  -> hourly pollution + weather readings
    predictions          -> what our ML model forecasted

NOTE: There is no "aqi" column on purpose. AQI is calculated from PM2.5/PM10,
so predicting it from those same values would cause target leakage.
Calculate the AQI category in the API layer when displaying results.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    """Current time in UTC (used as a default value for timestamps)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Location(Base):
    """A place we collect air-quality data for (e.g. 'Lahore')."""

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships let us write location.records instead of manual joins.
    # cascade: deleting a location also deletes its records/predictions.
    records: Mapped[List["AirQualityRecord"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        back_populates="location", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Location id={self.id} name={self.name!r}>"


class AirQualityRecord(Base):
    """One hourly reading: pollutants (µg/m³, CO in mg/m³) + weather."""

    __tablename__ = "air_quality_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Pollutants (nullable: real datasets always have gaps).
    pm25: Mapped[Optional[float]] = mapped_column(Float)
    pm10: Mapped[Optional[float]] = mapped_column(Float)
    no2: Mapped[Optional[float]] = mapped_column(Float)
    co: Mapped[Optional[float]] = mapped_column(Float)

    # Weather, stored in the same row so no joins are needed for training.
    temperature: Mapped[Optional[float]] = mapped_column(Float)  # °C
    humidity: Mapped[Optional[float]] = mapped_column(Float)  # %
    wind_speed: Mapped[Optional[float]] = mapped_column(Float)  # m/s

    location: Mapped["Location"] = relationship(back_populates="records")

    # One reading per location per hour. This also creates an index on
    # (location_id, timestamp), which makes time-range queries fast.
    __table_args__ = (
        UniqueConstraint("location_id", "timestamp", name="uq_location_timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<AirQualityRecord loc={self.location_id} "
            f"ts={self.timestamp} pm25={self.pm25}>"
        )


class Prediction(Base):
    """A forecast produced by the ML model (e.g. PM2.5 for 24h from now)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    location_id: Mapped[int] = mapped_column(
        ForeignKey("locations.id"), nullable=False
    )

    # When the prediction was made vs. the time it is predicting FOR.
    # Keeping both lets us later compare predicted vs actual values.
    made_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now
    )
    target_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    predicted_pm25: Mapped[float] = mapped_column(Float, nullable=False)

    # Which model produced it (e.g. "random_forest_v1"). Handy for comparisons.
    model_name: Mapped[Optional[str]] = mapped_column(String(50))

    location: Mapped["Location"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction loc={self.location_id} "
            f"target={self.target_time} pm25={self.predicted_pm25}>"
        )
