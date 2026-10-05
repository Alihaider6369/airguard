"""
Turn a PM2.5 value into an AQI number, a category, a colour and short advice.

Breakpoints: US EPA PM2.5 table, in force since 6 May 2024.

NOTE (be honest when presenting): the official AQI uses a 24-hour AVERAGE of
PM2.5. We apply the same table to a single hourly value, so what we show is
an "AQI-style" indicator, not an official AQI.
"""

import math
from typing import Dict

# (PM2.5 low, PM2.5 high, AQI low, AQI high, category, colour, advice)
_BREAKPOINTS = [
    (0.0, 9.0, 0, 50, "Good", "#00e400",
     "Air quality is satisfactory. Enjoy outdoor activities."),
    (9.1, 35.4, 51, 100, "Moderate", "#ffd400",
     "Acceptable. Unusually sensitive people should limit long outdoor effort."),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups", "#ff7e00",
     "Children, older adults and people with asthma or heart/lung conditions "
     "should reduce prolonged outdoor activity."),
    (55.5, 125.4, 151, 200, "Unhealthy", "#ff0000",
     "Everyone should reduce prolonged outdoor exertion. Consider a mask outdoors."),
    (125.5, 225.4, 201, 300, "Very Unhealthy", "#8f3f97",
     "Avoid outdoor activity. Keep windows closed and use a mask if you must go out."),
    (225.5, 325.4, 301, 500, "Hazardous", "#7e0023",
     "Health emergency conditions. Stay indoors and avoid all outdoor exertion."),
]


def describe_pm25(pm25: float) -> Dict[str, object]:
    """Return {'aqi', 'category', 'color', 'advice'} for a PM2.5 value (µg/m³)."""
    # EPA truncates PM2.5 to one decimal place before using the table.
    value = max(0.0, math.floor(pm25 * 10) / 10)

    for low, high, aqi_low, aqi_high, category, color, advice in _BREAKPOINTS:
        if value <= high:
            # Straight-line interpolation inside the band (EPA formula).
            aqi = (aqi_high - aqi_low) / (high - low) * (value - low) + aqi_low
            return {"aqi": round(aqi), "category": category, "color": color, "advice": advice}

    # Above the top of the table: cap at 500 / Hazardous.
    *_, category, color, advice = _BREAKPOINTS[-1]
    return {"aqi": 500, "category": category, "color": color, "advice": advice}
