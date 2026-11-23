"""Walk-forward backtest with as-of data (strategy section 1.5, "tester sans se mentir").

* Each day the model is re-run with ONLY the data published on or before that day
  (bitemporal store) and with betas/intercept estimated on the past only.
* Entry the next day (configurable delay), realistic costs: commission, 1 tick of
  slippage per side (2 ticks on report days).
* Exits: catastrophe stop (intraday high/low), dynamic targets, thesis and time stops.
* Robustness grid: z threshold 1.5 / 2 / 2.5 and entry delay 0 / 1 day.

Known limits (documented in the README): historical USDA balance sheets are final
values (PSD), not the vintages known at the time, and historical report dates are
approximated (WASDE ~ 10th of the month).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decision import AccountState, atr, decide
from .models.grains import analyze_grain


class MemoryStore:
    """Read-only in-memory copy of the observations table with fast as-of queries."""

    def __init__(self, store):
        with store.connect() as con:
            df = pd.read_sql_query("SELECT series_id, period, value, published_at FROM observations", con)
        df["period"] = pd.to_datetime(df["period"])
        df["published_at"] = pd.to_datetime(df["published_at"])
        self._data = {k: g.sort_values(["period", "published_at"]) for k, g in df.groupby("series_id")}
        self._cache: dict = {}

    def get_series(self, series_id: str, as_of=None) -> pd.Series:
        g = self._data.get(series_id)
        if g is None:
            return pd.Series(dtype=float, name=series_id)
        if as_of is not None:
            g = g[g["published_at"] <= pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)]
        g = g.groupby("period").tail(1)
        return pd.Series(g["value"].values, index=pd.DatetimeIndex(g["period"].values), name=series_id)

    def read_table(self, name):  # not used by models
        return pd.DataFrame()


def approx_report_calendar(start, end) -> pd.DataFrame:
    """Historical catalysts approximated: WASDE ~ business day nearest the 10th,
    Grain Stocks end of Mar/Jun/Sep and ~Jan 12, Acreage/Prospective Plantings."""
    rows = []
    for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        d = m + pd.Timedelta(days=9)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        rows.append({"date": d, "event": "WASDE", "markets": "corn|soybeans|wheat"})
        if m.month in (3, 6, 9):
            e = m + pd.offsets.BMonthEnd(0)
            rows.append({"date": e, "event": "GRAIN_STOCKS", "markets": "corn|soybeans|wheat"})
            if m.month == 3:
                rows.append({"date": e, "event": "PROSPECTIVE_PLANTINGS", "markets": "corn|soybeans|wheat"})
            if m.month == 6:
                rows.append({"date": e, "event": "ACREAGE", "markets": "corn|soybeans|wheat"})
        if m.month == 1:
            rows.append({"date": d + pd.Timedelta(days=2), "event": "GRAIN_STOCKS", "markets": "corn|soybeans|wheat"})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@dataclass
class BTConfig:
    z_entry: float = 2.0
    delay_days: int = 0   # 0 = enter at the next day's close; 1 = one more day (robustness)
    commission_per_contract_side: float = 1.0
    slippage_ticks: float = 1.0
    report_day_slippage_ticks: float = 2.0
    capital: float = 50000.0


def run_backtest(store, settings, market: str, start, end, cfg: BTConfig | None = None,
                 every: int = 1, progress: bool = True, _cache: dict | None = None) -> dict:
    cfg = cfg or BTConfig(capital=settings.account["capital_usd"])
    mem = store if isinstance(store, MemoryStore) else MemoryStore(store)
    s2 = copy.deepcopy(settings)
    s2.raw["signal"]["z_entry"] = cfg.z_entry
    s2.raw["markets"][market]["enabled"] = True
    mcfg = s2.markets[market]
    micro = mcfg["contracts"].get("micro") or mcfg["contracts"]["standard"]
    pv, tick = micro["point_value_usd"], micro["tick"]
    sig = s2.signal

    bars_all = pd.DataFrame({c: mem.get_series(f"px_{market}_{c}") for c in ("close", "high", "low")}).dropna(subset=["close"])
    days = bars_all.loc[pd.Timestamp(start):pd.Timestamp(end)].index
    cal = approx_report_calendar(pd.Timestamp(start) - pd.Timedelta(days=60), pd.Timestamp(end) + pd.Timedelta(days=60))
    report_days = set(cal["date"].dt.normalize())

    cache = _cache if _cache is not None else {}
    equity = cfg.capital
    eq_curve, trades = [], []
    pos = None          # open position dict
    pending = None      # (execute_on_index, decision)

    for i, d in enumerate(days):
        bar = bars_all.loc[d]
        slip = (cfg.report_day_slippage_ticks if d in report_days else cfg.slippage_ticks) * tick

        # --- execute pending entry
        if pending is not None and i >= pending[0] and pos is None:
            dec = pending[1]
            direction = 1 if dec.signal == "BUY" else -1
            entry = bar["close"] + direction * slip
            qty = max(int(dec.risk_usd // (max(sig["catastrophe_atr_mult"] * dec.atr20, mcfg.get("jump_allowance", 0)) * pv)), 0)
            if qty > 0:
                pos = {"dir": direction, "entry": entry, "date": d, "qty0": qty, "qty": qty,
                       "stop": entry - direction * sig["catastrophe_atr_mult"] * dec.atr20,
                       "fv0": dec.fair_value, "gap0": dec.fair_value - entry, "z0": dec.z,
                       "risk": qty * abs(sig["catastrophe_atr_mult"] * dec.atr20) * pv,
                       "pnl": -qty * cfg.commission_per_contract_side, "t1": False, "t2": False,
                       "trail": None, "exits": []}
            pending = None

        # --- model for today (cached across robustness runs)
        an = None
        if i % every == 0 or pos is not None:
            key = (market, d)
            if key not in cache:
                try:
                    cache[key] = analyze_grain(mem, s2, market, d)
                except Exception:
                    cache[key] = None
            an = cache[key]

        # --- manage open position
        if pos is not None:
            dirn = pos["dir"]
            def close_qty(q, px, why):
                q = min(q, pos["qty"])
                pos["pnl"] += (px - pos["entry"]) * dirn * q * pv - q * cfg.commission_per_contract_side
                pos["qty"] -= q
                pos["exits"].append(why)
            # catastrophe stop
            if (dirn > 0 and bar["low"] <= pos["stop"]) or (dirn < 0 and bar["high"] >= pos["stop"]):
                close_qty(pos["qty"], pos["stop"] - dirn * slip, "catastrophe")
            elif an is not None and not np.isnan(an.fair_value):
                fv = an.fair_value
                gap_tot = fv - pos["entry"]
                t1 = pos["entry"] + sig["partial_exit_1_fraction_of_gap"] * gap_tot
                t2 = pos["entry"] + sig["partial_exit_2_fraction_of_gap"] * gap_tot
                hi, lo = bar["high"], bar["low"]
                reach = (lambda lvl: hi >= lvl) if dirn > 0 else (lambda lvl: lo <= lvl)
                if not pos["t1"] and gap_tot * dirn > 0 and reach(t1):
                    close_qty(max(1, int(round(pos["qty0"] * sig["partial_exit_1_size"]))), t1 - dirn * slip, "target1")
                    pos["t1"] = True
                if pos["qty"] > 0 and pos["t1"] and not pos["t2"] and gap_tot * dirn > 0 and reach(t2):
                    close_qty(max(1, int(round(pos["qty0"] * sig["partial_exit_2_size"]))), t2 - dirn * slip, "target2")
                    pos["t2"] = True
                a20 = atr(bars_all.loc[:d].tail(40), sig["atr_period"])
                if pos["qty"] > 0 and pos["t2"]:
                    lvl = bar["close"] - dirn * sig["trailing_atr_mult"] * a20
                    pos["trail"] = lvl if pos["trail"] is None else (max(pos["trail"], lvl) if dirn > 0 else min(pos["trail"], lvl))
                    if (dirn > 0 and bar["close"] <= pos["trail"]) or (dirn < 0 and bar["close"] >= pos["trail"]):
                        close_qty(pos["qty"], bar["close"] - dirn * slip, "trailing")
                if pos["qty"] > 0 and (pos["fv0"] - fv) * np.sign(pos["gap0"]) > sig["thesis_break_fraction"] * abs(pos["gap0"]):
                    close_qty(pos["qty"], bar["close"] - dirn * slip, "thesis")
                n_rep = cal[(cal["date"] > pos["date"]) & (cal["date"] <= d) &
                            cal["event"].isin(mcfg.get("major_reports", []))].shape[0]
                if pos["qty"] > 0 and n_rep >= sig["time_stop_catalysts"] and abs(fv - bar["close"]) >= 0.5 * abs(pos["gap0"]):
                    close_qty(pos["qty"], bar["close"] - dirn * slip, "time")
            if pos["qty"] <= 0:
                equity += pos["pnl"]
                trades.append({"entry_date": pos["date"], "exit_date": d, "direction": "LONG" if dirn > 0 else "SHORT",
                               "entry": pos["entry"], "qty": pos["qty0"], "z_entry": pos["z0"],
                               "pnl_usd": pos["pnl"], "risk_usd": pos["risk"],
                               "R": pos["pnl"] / pos["risk"] if pos["risk"] else np.nan,
                               "exits": ",".join(pos["exits"])})
                pos = None

        # --- new signal
        if pos is None and pending is None and an is not None:
            state = AccountState(capital=equity)
            dec = decide(an, s2, state, cal)
            if dec.signal in ("BUY", "SELL"):
                pending = (i + 1 + cfg.delay_days, dec)

        mtm = equity + ((bar["close"] - pos["entry"]) * pos["dir"] * pos["qty"] * pv + pos["pnl"] if pos else 0.0)
        eq_curve.append((d, mtm))
        if progress and i % 50 == 0:
            print(f"  {market} {d:%Y-%m-%d}  equity {mtm:,.0f}  trades {len(trades)}", flush=True)

    eq = pd.Series(dict(eq_curve))
    tr = pd.DataFrame(trades)
    from .performance import metrics
    m = metrics(eq, tr.assign(status="CLOSED", r_multiple=tr["R"] if len(tr) else [],
                              realized_pnl_usd=tr["pnl_usd"] if len(tr) else []) if len(tr) else
                pd.DataFrame(columns=["status"]))
    return {"equity": eq, "trades": tr, "metrics": m, "cache": cache}


def robustness(store, settings, market, start, end, every: int = 1) -> pd.DataFrame:
    mem = MemoryStore(store)
    cache: dict = {}
    rows = []
    for z in (1.5, 2.0, 2.5):
        for delay in (0, 1):
            r = run_backtest(mem, settings, market, start, end,
                             BTConfig(z_entry=z, delay_days=delay, capital=settings.account["capital_usd"]),
                             every=every, progress=False, _cache=cache)
            met = dict(zip(r["metrics"]["indicator"], r["metrics"]["value"])) if len(r["metrics"]) else {}
            rows.append({"z_entry": z, "delay_days": delay, "n_trades": len(r["trades"]), **met})
    return pd.DataFrame(rows)
