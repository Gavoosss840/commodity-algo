"""Trade journal and open-position management (strategy rule 5 and section 1.3).

Tables in the store
  trades : one row per trade (thesis, z, gap and fair value at entry, initial risk)
  fills  : every execution (OPEN / PARTIAL / CLOSE) with quantity and price

Trades get in from three places:
  1. the daily Excel fiche: fill "Executed" / "Fill price" / "Qty filled" on the
     Decision sheet, or "Exit qty" / "Exit price" on the Positions sheet;
     the next run ingests them (excel_io.ingest_user_inputs);
  2. Interactive Brokers executions whose orderRef is a suggestion_id
     (orders staged by the algo);
  3. the CLI: `commodity-algo journal open|close ...`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COLS = ["trade_id", "market", "contract", "instrument", "direction", "entry_date", "entry_price",
              "qty_initial", "point_value_usd", "fair_value_entry", "z_entry", "gap_entry",
              "catastrophe_stop", "risk_usd_initial", "thesis", "suggestion_id", "status",
              "exit_date", "realized_pnl_usd", "r_multiple", "catalysts_seen", "notes"]


def open_trade(store, settings, *, market: str, contract: str, direction: str, entry_date, entry_price: float,
               qty: int, fair_value: float, z: float, catastrophe_stop: float, thesis: str = "",
               suggestion_id: str = "", instrument: str = "FUTURE", notes: str = "") -> str:
    mcfg = settings.markets[market]
    pv = next((c["point_value_usd"] for c in mcfg["contracts"].values() if c["symbol"] == contract),
              list(mcfg["contracts"].values())[0]["point_value_usd"])
    d = 1 if direction.upper() in ("BUY", "LONG") else -1
    trade_id = f"{market}-{pd.Timestamp(entry_date):%Y%m%d}-{contract}-{'L' if d > 0 else 'S'}"
    risk = abs(entry_price - catastrophe_stop) * pv * qty
    row = {"trade_id": trade_id, "market": market, "contract": contract, "instrument": instrument,
           "direction": "LONG" if d > 0 else "SHORT", "entry_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
           "entry_price": entry_price, "qty_initial": qty, "point_value_usd": pv,
           "fair_value_entry": fair_value, "z_entry": z, "gap_entry": fair_value - entry_price,
           "catastrophe_stop": catastrophe_stop, "risk_usd_initial": risk, "thesis": thesis,
           "suggestion_id": suggestion_id, "status": "OPEN", "exit_date": None,
           "realized_pnl_usd": 0.0, "r_multiple": np.nan, "catalysts_seen": 0, "notes": notes}
    store.upsert_table("trades", pd.DataFrame([row]), keys=["trade_id"])
    add_fill(store, trade_id, entry_date, "OPEN", qty, entry_price)
    return trade_id


def add_fill(store, trade_id: str, date, action: str, qty: int, price: float, source: str = "manual"):
    fid = f"{trade_id}|{pd.Timestamp(date):%Y%m%d}|{action}|{qty}|{price}"
    store.upsert_table("fills", pd.DataFrame([{
        "fill_id": fid, "trade_id": trade_id, "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "action": action, "qty": int(qty), "price": float(price), "source": source}]), keys=["fill_id"])


def close_part(store, trade_id: str, date, qty: int, price: float, source: str = "manual"):
    t = store.read_table("trades")
    row = t[t["trade_id"] == trade_id]
    if row.empty:
        raise KeyError(trade_id)
    open_qty = trade_open_qty(store, trade_id)
    action = "CLOSE" if qty >= open_qty else "PARTIAL"
    add_fill(store, trade_id, date, action, min(qty, open_qty), price, source)
    recompute_trade(store, trade_id)


def trade_open_qty(store, trade_id: str) -> int:
    f = store.read_table("fills")
    if f.empty:
        return 0
    f = f[f["trade_id"] == trade_id]
    opened = f.loc[f["action"] == "OPEN", "qty"].astype(int).sum()
    closed = f.loc[f["action"] != "OPEN", "qty"].astype(int).sum()
    return int(opened - closed)


def recompute_trade(store, trade_id: str):
    t = store.read_table("trades")
    f = store.read_table("fills")
    row = t[t["trade_id"] == trade_id].iloc[0].to_dict()
    ff = f[f["trade_id"] == trade_id]
    d = 1 if row["direction"] == "LONG" else -1
    pv = float(row["point_value_usd"])
    entry = float(row["entry_price"])
    exits = ff[ff["action"] != "OPEN"]
    pnl = float(((exits["price"].astype(float) - entry) * d * exits["qty"].astype(int) * pv).sum())
    row["realized_pnl_usd"] = pnl
    risk = float(row["risk_usd_initial"]) or np.nan
    row["r_multiple"] = pnl / risk if risk else np.nan
    if trade_open_qty(store, trade_id) <= 0:
        row["status"] = "CLOSED"
        row["exit_date"] = exits["date"].max()
    store.upsert_table("trades", pd.DataFrame([row]), keys=["trade_id"])


def open_trades(store) -> pd.DataFrame:
    t = store.read_table("trades")
    return t[t["status"] == "OPEN"].copy() if not t.empty else t


def manage_positions(store, settings, analyses: dict, calendar: pd.DataFrame, asof) -> pd.DataFrame:
    """Daily review of each open trade: P&L, R, updated dynamic targets and
    which of the three stops (thesis / catastrophe / time) is triggered."""
    asof = pd.Timestamp(asof)
    sig = settings.signal
    rows = []
    for _, t in open_trades(store).iterrows():
        an = analyses.get(t["market"])
        d = 1 if t["direction"] == "LONG" else -1
        qty_open = trade_open_qty(store, t["trade_id"])
        entry, pv = float(t["entry_price"]), float(t["point_value_usd"])
        price = an.price if an is not None else np.nan
        fv = an.fair_value if an is not None else np.nan
        bars = an.bars if an is not None else None
        from .decision import atr
        a20 = atr(bars, sig["atr_period"]) if bars is not None else np.nan
        gap0 = float(t["gap_entry"])
        gap_now = fv - price if not np.isnan(fv) else np.nan
        unreal = (price - entry) * d * qty_open * pv if not np.isnan(price) else np.nan
        risk0 = float(t["risk_usd_initial"]) or np.nan
        realized = float(t["realized_pnl_usd"] or 0.0)
        actions = []
        # catastrophe stop
        cat = float(t["catastrophe_stop"])
        if bars is not None and len(bars):
            last = bars.iloc[-1]
            if (d > 0 and last["low"] <= cat) or (d < 0 and last["high"] >= cat):
                actions.append(f"CATASTROPHE STOP touched ({cat:g}) - check the IB fill")
        # thesis stop: fair value moved against us by > fraction of the entry gap
        fv0 = float(t["fair_value_entry"])
        if not np.isnan(fv) and gap0 != 0:
            moved_against = (fv0 - fv) * np.sign(gap0)
            if moved_against > sig["thesis_break_fraction"] * abs(gap0):
                actions.append(f"THESIS STOP: fair value {fv0:g} -> {fv:g} (moved against by > "
                               f"{int(100*sig['thesis_break_fraction'])}% of entry gap) - EXIT")
        # time stop: catalysts since entry
        mcfg = settings.markets[t["market"]]
        if not calendar.empty:
            seen = calendar[calendar["event"].isin(mcfg.get("major_reports", [])) &
                            calendar["markets"].fillna("").str.contains(t["market"]) &
                            (calendar["date"] > pd.Timestamp(t["entry_date"])) & (calendar["date"] <= asof)]
            n_seen = len(seen)
        else:
            n_seen = 0
        if n_seen >= sig["time_stop_catalysts"] and not np.isnan(gap_now) and abs(gap_now) >= 0.5 * abs(gap0):
            actions.append(f"TIME STOP: {n_seen} reports since entry and gap not reduced - EXIT")
        # dynamic targets
        t1 = entry + sig["partial_exit_1_fraction_of_gap"] * (gap_now + (price - entry)) if not np.isnan(gap_now) else np.nan
        t2 = entry + sig["partial_exit_2_fraction_of_gap"] * (gap_now + (price - entry)) if not np.isnan(gap_now) else np.nan
        closed_share = 1 - qty_open / max(int(t["qty_initial"]), 1)
        if not np.isnan(price):
            if closed_share < sig["partial_exit_1_size"] - 1e-9 and (price - t1) * d >= 0:
                actions.append(f"TARGET 1 reached ({t1:g}): take {int(100*sig['partial_exit_1_size'])}% off")
            elif closed_share < sig["partial_exit_1_size"] + sig["partial_exit_2_size"] - 1e-9 and (price - t2) * d >= 0:
                actions.append(f"TARGET 2 reached ({t2:g}): take another {int(100*sig['partial_exit_2_size'])}% off; "
                               f"trail the rest at {sig['trailing_atr_mult']} ATR")
        trail = price - d * sig["trailing_atr_mult"] * a20 if not np.isnan(a20) else np.nan
        rows.append({
            "trade_id": t["trade_id"], "market": t["market"], "contract": t["contract"],
            "direction": t["direction"], "entry_date": t["entry_date"], "entry_price": entry,
            "qty_open": qty_open, "price": price, "fair_value_now": fv, "fair_value_entry": fv0,
            "z_now": an.z if an is not None else np.nan, "unrealized_pnl_usd": unreal,
            "realized_pnl_usd": realized, "open_R": (unreal + realized) / risk0 if risk0 else np.nan,
            "catastrophe_stop": cat, "target_1": t1, "target_2": t2, "trailing_stop_if_active": trail,
            "risk_open_usd": abs(price - cat) * qty_open * pv if not np.isnan(price) else risk0,
            "group": mcfg.get("group", ""), "reports_since_entry": n_seen,
            "action": " | ".join(actions) if actions else "HOLD",
            "exit_qty": None, "exit_price": None,  # <- YOU fill these in Excel after exiting
        })
        store.upsert_table("trades", pd.DataFrame([{**t.to_dict(), "catalysts_seen": n_seen}]), keys=["trade_id"])
    return pd.DataFrame(rows)
