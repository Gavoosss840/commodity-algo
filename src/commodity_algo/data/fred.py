"""FRED macro series (free, no API key needed for the CSV endpoint).

Series used by the study:
  DTWEXBGS  broad trade-weighted US dollar index (daily)
  DFII10    10-year TIPS real yield (daily)
  VIXCLS    VIX (daily)
  DHHNGSP   Henry Hub natural gas spot (daily)
  DCOILWTICO WTI spot (daily) - fallback price for oil
  PCOPPUSDM copper, IMF monthly
  CPIAUCSL  CPI, to deflate prices into real terms (monthly)
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# Publication lag rules (approximate, used to timestamp history when backfilling)
_LAG = {
    "daily": pd.offsets.BDay(1),
    "monthly": pd.DateOffset(months=1, days=14),  # e.g. CPI for March known mid-April
}
SERIES_FREQ = {
    "DTWEXBGS": "daily", "DFII10": "daily", "VIXCLS": "daily", "DHHNGSP": "daily",
    "DCOILWTICO": "daily", "PCOPPUSDM": "monthly", "CPIAUCSL": "monthly",
}


def fetch_fred(series_id: str, start: str = "1990-01-01", timeout: int = 60) -> pd.Series:
    for attempt in range(4):
        try:
            r = requests.get(FRED_CSV.format(sid=series_id), timeout=timeout)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[series_id], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s = s.dropna()
    return s[s.index >= pd.Timestamp(start)].rename(series_id)


def publication_dates(series_id: str, s: pd.Series) -> pd.Series:
    lag = _LAG[SERIES_FREQ.get(series_id, "daily")]
    return pd.Series([d + lag for d in s.index], index=s.index)


def update_fred(store, series_ids, start: str = "1990-01-01") -> dict[str, int]:
    out = {}
    for sid in series_ids:
        s = fetch_fred(sid, start=start)
        out[sid] = store.upsert_observations(sid, s, publication_dates(sid, s), source="FRED")
    return out
