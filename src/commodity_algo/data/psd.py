"""USDA FAS PSD balance sheets (the data behind WASDE), via api.fas.usda.gov.

Free key from https://api.data.gov/signup/  -> USDA_FAS_API_KEY in .env
(DEMO_KEY works for testing but is rate-limited).

Stored series, one value per marketing year, period = <MY>-05-01 (the month the
WASDE starts projecting that crop year):
  <market>_psd_area, _yield, _beginning_stocks, _production, _imports,
  _domestic_use, _exports, _ending_stocks, _total_use, _stocks_to_use
  corn_china_imports, soybeans_brazil_production, wheat_world_ex_us_china_su

Bitemporal handling
  * Current and previous marketing year: published_at = the PSD vintage month
    (true vintages; one new row per WASDE as the algo runs month after month).
  * Older years (backfill): final revised values stamped at <MY>-05-10.
    This is a KNOWN look-ahead approximation for historical calibration only.
    A clean backtest needs the WASDE archive vintages (see README roadmap).
"""
from __future__ import annotations

import time

import pandas as pd
import requests

BASE = "https://api.fas.usda.gov/api/psd"
CODES = {"corn": "0440000", "soybeans": "2222000", "wheat": "0410000"}
ATTR = {4: "area", 20: "beginning_stocks", 28: "production", 57: "imports",
        88: "exports", 125: "domestic_use", 176: "ending_stocks", 184: "yield"}
MT_TO_MBU = {"corn": 39.368 / 1000.0, "soybeans": 36.744 / 1000.0, "wheat": 36.744 / 1000.0}


def _get(path: str, key: str, retries: int = 4):
    for i in range(retries):
        r = requests.get(f"{BASE}/{path}", params={"api_key": key}, timeout=60)
        if r.status_code == 429 and i < retries - 1:
            time.sleep(15 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    return []


def _balance(rows: list[dict]) -> tuple[dict, pd.Timestamp | None]:
    vals, vintage = {}, None
    for r in rows:
        name = ATTR.get(r["attributeId"])
        if name:
            vals[name] = float(r["value"])
        if r.get("calendarYear") and r.get("month"):
            vintage = pd.Timestamp(int(r["calendarYear"]), int(r["month"]), 12)
    return vals, vintage


def fetch_year(market: str, year: int, key: str, region: str = "country/US"):
    return _balance(_get(f"commodity/{CODES[market]}/{region}/year/{year}", key))


def update_psd(store, market: str, key: str | None, first_year: int = 1992,
               asof: pd.Timestamp | None = None, full: bool = False) -> dict:
    if not key:
        return {"skipped": "USDA_FAS_API_KEY missing (DEMO_KEY works for a test)"}
    asof = pd.Timestamp(asof or pd.Timestamp.today())
    cur_my = asof.year if asof.month >= 5 else asof.year - 1
    existing = store.get_series(f"{market}_psd_stocks_to_use")
    years = range(first_year, cur_my + 1) if (full or existing.empty) else range(cur_my - 1, cur_my + 1)
    k = MT_TO_MBU[market]
    n = 0
    for my in years:
        bal, vintage = fetch_year(market, my, key)
        if not bal or not bal.get("ending_stocks"):
            continue
        total_use = bal.get("domestic_use", 0.0) + bal.get("exports", 0.0)
        derived = {
            "area": bal.get("area", float("nan")) * 2.47105 / 1000.0,       # 1000 ha -> million acres
            "yield": bal.get("yield", float("nan")) * k * 1000.0 / 2.47105,  # t/ha -> bu/acre
            "beginning_stocks": bal.get("beginning_stocks", 0.0) * k,
            "production": bal.get("production", 0.0) * k,
            "imports": bal.get("imports", 0.0) * k,
            "domestic_use": bal.get("domestic_use", 0.0) * k,
            "exports": bal.get("exports", 0.0) * k,
            "ending_stocks": bal["ending_stocks"] * k,
            "total_use": total_use * k,
            "stocks_to_use": bal["ending_stocks"] / total_use if total_use else float("nan"),
        }
        period = pd.Timestamp(my, 5, 1)
        recent = my >= cur_my - 1
        pub = vintage if (recent and vintage is not None) else pd.Timestamp(my, 5, 10)
        for name, v in derived.items():
            n += store.upsert_observations(f"{market}_psd_{name}", pd.Series({period: v}),
                                           pub.strftime("%Y-%m-%d"), source="USDA PSD")
        # market-specific extra factors
        if market == "corn":
            ch, _ = fetch_year(market, my, key, "country/CH")
            if ch.get("imports"):
                n += store.upsert_observations("corn_china_imports", pd.Series({period: ch["imports"] / 1000.0}),
                                               pub.strftime("%Y-%m-%d"), source="USDA PSD")
        elif market == "soybeans":
            br, _ = fetch_year(market, my, key, "country/BR")
            if br.get("production"):
                n += store.upsert_observations("soybeans_brazil_production",
                                               pd.Series({period: br["production"] / 1000.0}),
                                               pub.strftime("%Y-%m-%d"), source="USDA PSD")
        elif market == "wheat":
            w, _ = fetch_year(market, my, key, "world")
            ch, _ = fetch_year(market, my, key, "country/CH")
            stocks = w.get("ending_stocks", 0) - bal.get("ending_stocks", 0) - ch.get("ending_stocks", 0)
            use = (w.get("domestic_use", 0) - bal.get("domestic_use", 0) - ch.get("domestic_use", 0))
            if use > 0:
                n += store.upsert_observations("wheat_world_ex_us_china_su", pd.Series({period: stocks / use}),
                                               pub.strftime("%Y-%m-%d"), source="USDA PSD")
    return {"rows": n}
