"""Manual inputs that you update after each official report.

input/wasde_balance.csv  - one row per (report_date, market): the USDA balance sheet
                           published that day. Takes ~2 minutes after each WASDE.
input/calendar.csv       - dated catalysts (WASDE, Grain Stocks, EIA, OPEC...).

Each WASDE row is stored with published_at = report_date, so the model only
"knows" a balance sheet from the day it was released (no look-ahead).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WASDE_COLUMNS = [
    "report_date", "market", "marketing_year", "area_harvested_macres", "usda_yield",
    "beginning_stocks", "production", "imports", "total_use", "exports", "ending_stocks",
    "china_imports", "brazil_production", "world_ex_us_china_stocks",
    "world_ex_us_china_use", "notes",
]


def read_wasde(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=WASDE_COLUMNS)
    df = pd.read_csv(path, comment="#")
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.sort_values("report_date")


def validate_wasde(df: pd.DataFrame, market: str) -> list[str]:
    """Return human-readable problems for the latest row of a market."""
    sub = df[df["market"] == market]
    if sub.empty:
        return [f"{market}: no WASDE row in input/wasde_balance.csv"]
    last = sub.iloc[-1]
    need = ["area_harvested_macres", "usda_yield", "total_use", "ending_stocks"]
    if market == "wheat":
        need = ["world_ex_us_china_stocks", "world_ex_us_china_use"]
    missing = [c for c in need if pd.isna(last.get(c))]
    return [f"{market}: WASDE {last['report_date'].date()} missing {', '.join(missing)}"] if missing else []


def store_wasde(store, df: pd.DataFrame) -> int:
    n = 0
    numeric = [c for c in WASDE_COLUMNS if c not in ("report_date", "market", "notes", "marketing_year")]
    for market, sub in df.groupby("market"):
        for col in numeric:
            if col not in sub:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce")
            s = pd.Series(vals.values, index=sub["report_date"].values).dropna()
            if s.empty:
                continue
            s = s[~s.index.duplicated(keep="last")]
            pub = pd.Series(s.index, index=s.index)
            n += store.upsert_observations(f"{market}_{col}", s, pub, source="USDA WASDE (manual)")
    return n


def read_calendar(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["date", "event", "markets", "time_et", "verified"])
    df = pd.read_csv(path, comment="#")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def next_catalysts(cal: pd.DataFrame, market: str, events: list[str], asof, n: int = 3) -> pd.DataFrame:
    asof = pd.Timestamp(asof).normalize()
    m = cal["event"].isin(events) & cal["markets"].fillna("").str.contains(market) & (cal["date"] >= asof)
    return cal[m].head(n)


def business_days_between(a, b) -> int:
    return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(b).date()))
