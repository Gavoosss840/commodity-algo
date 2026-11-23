"""Daily weather for the producing regions (Open-Meteo: ERA5 archive + 16-day forecast).

Open-Meteo is free WITHOUT a key for non-commercial use. For commercial use set
OPEN_METEO_API_KEY in .env (paid plan) and the customer endpoints are used.

Stored series (per weather point, e.g. "Iowa"):
  wx_<point>_tmax   daily max temperature, deg F (observed, ERA5)
  wx_<point>_tmin   daily min temperature, deg F
  wx_<point>_prcp   daily precipitation, inches
  wxf_<point>_*     same variables, FORECAST vintage (published_at = fetch date)
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

VARS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "prcp"}


def _hosts():
    key = os.environ.get("OPEN_METEO_API_KEY")
    if key:
        return ("https://customer-archive-api.open-meteo.com/v1/archive",
                "https://customer-api.open-meteo.com/v1/forecast", key)
    return ("https://archive-api.open-meteo.com/v1/archive",
            "https://api.open-meteo.com/v1/forecast", None)


def _get(url: str, params: dict, retries: int = 4) -> dict:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429 and i < retries - 1:
            time.sleep(20 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("unreachable")


def _to_frame(js: dict) -> pd.DataFrame:
    d = js["daily"]
    df = pd.DataFrame({VARS[k]: d[k] for k in VARS}, index=pd.to_datetime(d["time"]))
    return df.astype(float)


def fetch_archive(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url, _, key = _hosts()
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": ",".join(VARS), "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch", "timezone": "America/Chicago",
    }
    if key:
        params["apikey"] = key
    return _to_frame(_get(url, params))


def fetch_forecast(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    _, url, key = _hosts()
    params = {
        "latitude": lat, "longitude": lon, "forecast_days": days,
        "daily": ",".join(VARS), "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch", "timezone": "America/Chicago",
    }
    if key:
        params["apikey"] = key
    return _to_frame(_get(url, params))


def update_weather(store, points: list[dict], start: str = "1960-01-01") -> dict[str, int]:
    """Incremental: only fetches the archive after the last stored date."""
    today = pd.Timestamp.today().normalize()
    archive_end = (today - pd.Timedelta(days=6)).strftime("%Y-%m-%d")  # ERA5 delay ~5 days
    out = {}
    for p in points:
        name = p["name"]
        existing = store.get_series(f"wx_{name}_tmax")
        s0 = (existing.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if len(existing) else start
        if s0 <= archive_end:
            df = fetch_archive(p["lat"], p["lon"], s0, archive_end)
            pub = pd.Series([d + pd.Timedelta(days=6) for d in df.index], index=df.index)
            for col in df.columns:
                out[f"wx_{name}_{col}"] = store.upsert_observations(
                    f"wx_{name}_{col}", df[col], pub, source="Open-Meteo ERA5")
        fc = fetch_forecast(p["lat"], p["lon"])
        fc = fc[fc.index > pd.Timestamp(archive_end)]
        for col in fc.columns:
            out[f"wxf_{name}_{col}"] = store.upsert_observations(
                f"wxf_{name}_{col}", fc[col], today.strftime("%Y-%m-%d"), source="Open-Meteo forecast")
    return out
