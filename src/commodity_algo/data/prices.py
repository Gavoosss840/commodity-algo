"""Futures prices: Interactive Brokers first, Yahoo Finance as a fallback.

Stored observation series per market:
  px_<market>_close / _high / _low / _volume   continuous front-month (model calibration)
  px_<market>_next_close                        second contract (carry filter)
Each run also records the exact contract being traded in the 'contracts' table.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def _store_bars(store, market: str, df: pd.DataFrame, prefix: str, source: str) -> int:
    if df is None or df.empty:
        return 0
    n = 0
    pub = pd.Series(df.index, index=df.index)  # a close is known the same evening
    for col in ("close", "high", "low", "volume"):
        if col in df:
            n += store.upsert_observations(f"{prefix}_{col}", df[col].astype(float), pub, source=source)
    return n


def price_scale(df: pd.DataFrame, mcfg: dict) -> float:
    """Grain models work in cents/bu. If a source quotes USD/bu (close < 50), scale x100."""
    if df is None or df.empty or "cents" not in mcfg.get("price_unit", ""):
        return 1.0
    return 100.0 if float(df["close"].median()) < 50 else 1.0


def _scaled(df: pd.DataFrame, k: float) -> pd.DataFrame:
    if df is None or df.empty or k == 1.0:
        return df
    df = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in df:
            df[c] = df[c] * k
    return df


def update_prices_ib(store, ib, market: str, mcfg: dict, years: int = 10) -> dict:
    spec = mcfg["contracts"].get("standard") or mcfg["contracts"]["micro"]
    out = {}
    cont = ib.continuous_bars(spec["symbol"], spec["exchange"], duration=f"{years} Y")
    k = price_scale(cont, mcfg)
    out["ib_price_scale"] = k
    cont = _scaled(cont, k)
    out["continuous"] = _store_bars(store, market, cont, f"px_{market}", "IB continuous")

    # Contract actually traded: micro if it exists (small capital), else standard
    contracts = {}
    for kind, c in mcfg["contracts"].items():
        try:
            front, nxt = ib.active_contracts(c["symbol"], c["exchange"],
                                             mcfg.get("roll_business_days_before_expiry", 10))
        except Exception as e:
            log.warning("%s %s: %s", market, kind, e)
            continue
        bars = _scaled(ib.daily_bars(front, duration="1 Y"), k)
        contracts[kind] = {"info": front, "bars": bars, "next": nxt}
        if kind == "standard" and nxt is not None:
            nb = _scaled(ib.daily_bars(nxt, duration="1 Y"), k)
            out["next"] = _store_bars(store, market, nb[["close"]], f"px_{market}_next", "IB")
    store.upsert_table("contracts", pd.DataFrame([
        {"run_date": pd.Timestamp.today().strftime("%Y-%m-%d"), "market": market, "kind": kd,
         "local_symbol": v["info"].local_symbol, "expiry": v["info"].expiry.strftime("%Y-%m-%d"),
         "con_id": v["info"].con_id, "last_close": float(v["bars"]["close"].iloc[-1]) if len(v["bars"]) else None,
         "avg_volume_20d": float(v["bars"]["volume"].tail(20).mean()) if len(v["bars"]) else None,
         "next_local_symbol": v["next"].local_symbol if v["next"] is not None else None}
        for kd, v in contracts.items()]), keys=["run_date", "market", "kind"])
    out["contracts"] = contracts
    return out


def update_prices_yahoo(store, market: str, mcfg: dict, years: int = 10) -> dict:
    """Fallback when IB is not running. Yahoo continuous front month, cents for grains."""
    import yfinance as yf
    sym = mcfg.get("yahoo_symbol")
    if not sym:
        return {}
    df = yf.download(sym, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return {"continuous": 0}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _scaled(df, price_scale(df, mcfg))
    return {"continuous": _store_bars(store, market, df, f"px_{market}", "Yahoo")}


def load_bars(store, market: str, as_of=None) -> pd.DataFrame:
    cols = {c: store.get_series(f"px_{market}_{c}", as_of) for c in ("close", "high", "low", "volume")}
    df = pd.DataFrame(cols).dropna(subset=["close"])
    nxt = store.get_series(f"px_{market}_next_close", as_of)
    if len(nxt):
        df["next_close"] = nxt.reindex(df.index)
    return df
