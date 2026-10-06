"""
History endpoint: daily averages from the database.

GET /api/history?city=Lahore&days=7

The window ends at the NEWEST record stored for that city (not "today"), so
the chart still works when the database holds older data from CSV files.
AQI is calculated here from the daily PM2.5 average (the models.py note says
AQI is deliberately not stored).
"""

from collections import defaultdict
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db

router = APIRouter(prefix="/api", tags=["history"])

# US EPA PM2.5 breakpoints: (conc_low, conc_high, aqi_low, aqi_high)
_PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


def _aqi_from_pm25(conc: Optional[float]) -> Optional[int]:
    if conc is None:
        return None
    conc = int(conc * 10) / 10  # truncate to 1 decimal like the EPA formula
    for lo, hi, a_lo, a_hi in _PM25_BREAKPOINTS:
        if conc <= hi:
            return round((a_hi - a_lo) / (hi - lo) * (conc - lo) + a_lo)
    return 500


def _avg(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else None


@router.get("/history")
def get_history(
    city: str = Query(..., min_length=2),
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
):
    location = crud.get_location_by_name(db, city)
    if location is None:
        raise HTTPException(status_code=404, detail=f"No data stored for '{city}'.")

    newest = crud.get_latest_records(db, location.id, limit=1)
    if not newest:
        return {"city": location.name, "days": days, "last_updated": None, "points": []}

    end = newest[-1].timestamp
    records = crud.get_records(db, location.id, start=end - timedelta(days=days), end=end)

    by_day = defaultdict(list)
    for rec in records:
        by_day[rec.timestamp.date()].append(rec)

    points = []
    for day in sorted(by_day):
        rows = by_day[day]
        pm25 = _avg([r.pm25 for r in rows])
        points.append(
            {
                "date": day.isoformat(),
                "pm25": pm25,
                "pm10": _avg([r.pm10 for r in rows]),
                "no2": _avg([r.no2 for r in rows]),
                "co": _avg([r.co for r in rows]),
                "aqi": _aqi_from_pm25(pm25),
            }
        )

    return {
        "city": location.name,
        "days": days,
        "last_updated": end.isoformat(),
        "points": points,
    }
