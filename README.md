# AirGuard – AI Air Pollution Prediction System for Pakistan

**Live demo:** https://alihaider63.pythonanywhere.com/

AirGuard shows live air quality for Pakistani cities, predicts tomorrow's PM2.5
with a machine-learning model, and explains what is causing the pollution and
how to stay safe.

## Sustainable Development Goals
- **SDG 3 – Good Health and Well-being:** warns people about unhealthy air and gives health precautions.
- **SDG 11 – Sustainable Cities and Communities:** tracks air quality across major cities and shows pollution causes.
- **SDG 13 – Climate Action:** explains pollution sources (fuel burning, dust, industry) and how to reduce them.

## Features
- Live AQI, PM2.5, PM10, NO₂ and CO for Pakistani cities
- 24-hour PM2.5 forecast with typical model error
- AQI history, map, alerts, causes and precautions
- "My location" to see the nearest air-quality data

## Tech stack
FastAPI, SQLAlchemy + SQLite, scikit-learn (joblib model), Tailwind CSS, Chart.js, Leaflet.js. Hosted on PythonAnywhere.

---
# AirGuard - Live Air Quality & 24-Hour PM2.5 Forecast for Any City in Pakistan

Type any Pakistani city or town; AirGuard shows the current air quality, the last
7 days, and a machine-learning forecast of PM2.5 for the next 24 hours.

Related SDGs: 3 (Good Health), 11 (Sustainable Cities), 13 (Climate Action).

## How it works for ANY city

```
city name --> geocoding --> coordinates --> live data (last 7 days) --> pooled ML model --> forecast
 "Multan"    Open-Meteo       30.2, 71.5      Open-Meteo (cached)       trained on 15 cities
```

* The model is trained once on years of history from 15 cities spread across the
  country, then applied to any place. Its accuracy on cities it **never saw** is
  measured and shown by `train_model.py` (the "UNSEEN cities" table).
* No database lookup is needed for a new city, so nothing has to be pre-loaded.

The page at `/` has a **Use my location** button and detects the location
automatically once the browser permission is on (works on https or localhost).
GPS coordinates are rounded to ~1 km and never stored.

## Important note about the data

Pollution values come from the **CAMS atmospheric model** (via the free
[Open-Meteo](https://open-meteo.com) API) on a coarse grid (~45 km). They are model
estimates, **not ground-sensor readings**. Weather comes from ERA5 (training) and
Open-Meteo forecast models (live), a small mismatch we acknowledge. The AQI shown
is an "AQI-style" indicator: it applies the US EPA PM2.5 table (2024) to an hourly
value, whereas the official AQI uses a 24-hour average.
Open-Meteo's free API is for non-commercial use; credit Open-Meteo and CAMS.

## Project structure

```
app/
  main.py           FastAPI routes
  schemas.py        JSON response shapes
  aqi.py            PM2.5 -> AQI category / colour / advice
  config.py         settings (.env)
  database.py       SQLAlchemy engine + sessions
  models.py         tables: locations, air_quality_records, predictions
  crud.py           reusable database functions
  services/
    open_meteo.py   shared download helpers
    geocoding.py    city name -> coordinates (Pakistan only)
    live_data.py    latest 7 days of data, cached 30 minutes
  ml/
    features.py     feature engineering (shared by training + API)
    predict.py      load the model and forecast
scripts/
  init_db.py            create tables
  fetch_open_meteo.py   download years of training history (15 cities)
  train_model.py        train + evaluate + save the pooled model
  check_db.py           row counts
  load_csv.py           load any CSV (optional)
  generate_sample_data.py   FAKE data, testing only
```

## Setup

```bash
pip install -r requirements.txt

python -m scripts.init_db
python -m scripts.fetch_open_meteo       # ~15 cities, takes several minutes
python -m scripts.check_db
python -m scripts.train_model            # saves models/ and reports/

uvicorn app.main:app --reload            # open http://127.0.0.1:8000/docs
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | server check |
| `GET /api/places/search?q=mult` | find Pakistani cities by name (search box) |
| `GET /api/places/reverse?lat=..&lon=..` | which Pakistani place a GPS position is in |
| any endpoint with `lat=..&lon=..` instead of `city=` | use the visitor's GPS location |
| `GET /api/dashboard?city=Multan` | current + 7-day history + forecast in one call |
| `GET /api/air-quality/current?city=Multan` | newest reading + AQI status |
| `GET /api/air-quality/history?city=Multan&hours=168` | hourly data for charts |
| `GET /api/forecast?city=Multan` | PM2.5 forecast, 24 hours ahead |

## Model evaluation

Ridge, Random Forest and Gradient Boosting are compared with a "same as now"
baseline on a **time-based** split (older 80% train, newest 20% test, 24 h gap).
The best model is also retrained without two cities (default Quetta, Sukkur) and
tested on exactly those, to show it works for places it was not trained on.

## Switching to PostgreSQL

Set `DATABASE_URL` in `.env`, install `psycopg2-binary`, run `init_db` again.
