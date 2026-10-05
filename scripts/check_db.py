"""
Quick health check: shows how much data is in the database.

Run from the project root:
    python -m scripts.check_db
"""

from sqlalchemy import func, select

from app.database import SessionLocal
from app.models import AirQualityRecord, Location, Prediction


def main() -> None:
    with SessionLocal() as db:
        print("Locations :", db.scalar(select(func.count(Location.id))))
        print("Records   :", db.scalar(select(func.count(AirQualityRecord.id))))
        print("Predictions:", db.scalar(select(func.count(Prediction.id))))

        # Per-location summary: row count and the time range covered.
        rows = db.execute(
            select(
                Location.name,
                func.count(AirQualityRecord.id),
                func.min(AirQualityRecord.timestamp),
                func.max(AirQualityRecord.timestamp),
            )
            .join(AirQualityRecord, AirQualityRecord.location_id == Location.id)
            .group_by(Location.name)
        ).all()
        for name, count, first, last in rows:
            print(f"  {name}: {count} rows, {first} -> {last}")


if __name__ == "__main__":
    main()
