"""EIA weekly petroleum data (free API key: https://www.eia.gov/opendata/register.php).

Set EIA_API_KEY in .env. Only needed for the WTI model (roadmap stage 4).
"""
from __future__ import annotations

import pandas as pd
import requests

URL = "https://api.eia.gov/v2/seriesid/{sid}"

SERIES = {
    "eia_crude_stocks": "PET.WCESTUS1.W",              # US crude ex-SPR, thousand bbl
    "eia_cushing_stocks": "PET.W_EPC0_SAX_YCUOK_MBBL.W",  # Cushing OK, thousand bbl
    "eia_us_production": "PET.WCRFPUS2.W",             # US field production, kb/d
}


def fetch_eia(series_code: str, api_key: str) -> pd.Series:
    r = requests.get(URL.format(sid=series_code), params={"api_key": api_key}, timeout=60)
    r.raise_for_status()
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.to_datetime(df["period"])
    return s.sort_index().dropna()


def update_eia(store, api_key: str | None) -> dict[str, int]:
    if not api_key:
        return {"skipped": "EIA_API_KEY missing"}
    out = {}
    for sid, code in SERIES.items():
        s = fetch_eia(code, api_key)
        if "stocks" in sid:
            s = s / 1000.0  # -> million barrels
        # week ending Friday, released the following Wednesday 10:30 ET
        pub = pd.Series([d + pd.Timedelta(days=5) for d in s.index], index=s.index)
        out[sid] = store.upsert_observations(sid, s, pub, source="EIA")
    return out
