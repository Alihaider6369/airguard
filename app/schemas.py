"""
Pydantic schemas = the exact JSON shapes the API returns.
FastAPI uses them to validate responses and to build the /docs page.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class PlaceOut(BaseModel):
    """A Pakistani city/town found by name."""
    name: str
    region: Optional[str] = None        # province / admin area
    display_name: str
    latitude: float
    longitude: float


class ReadingOut(BaseModel):
    """One hourly measurement."""
    timestamp: datetime
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    co: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None


class StatusOut(BaseModel):
    """AQI-style summary of a PM2.5 value (see app/aqi.py)."""
    aqi: int
    category: str
    color: str
    advice: str


class CurrentOut(BaseModel):
    place: PlaceOut
    reading: ReadingOut
    status: StatusOut
    data_age_hours: float  # how old the newest reading is


class HistoryOut(BaseModel):
    place: PlaceOut
    hours: int
    readings: List[ReadingOut]


class ForecastOut(BaseModel):
    place: PlaceOut
    as_of: datetime            # time of the newest data used
    target_time: datetime      # time being forecast (as_of + 24h)
    current_pm25: float
    predicted_pm25: float
    typical_error: Optional[float] = None  # model's average error on unseen test data (µg/m³)
    trend: str                 # "rising" | "falling" | "stable"
    status: StatusOut          # status of the PREDICTED value
    model_name: str
    note: str


class DashboardOut(BaseModel):
    """Everything the dashboard page needs, in one request."""
    place: PlaceOut
    current: CurrentOut
    history: List[ReadingOut]
    forecast: ForecastOut
