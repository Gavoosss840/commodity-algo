"""Equity tracking, drawdown state and the gate-B indicators (strategy section 1.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

GATE_B = {"sharpe_min": 0.7, "max_dd_max_pct": 15.0, "expectancy_R_min": 0.2, "corr_abs_max": 0.3}


def record_equity(store, asof, equity: float, source: str):
    store.upsert_table("equity", pd.DataFrame([{
        "date": pd.Timestamp(asof).strftime("%Y-%m-%d"), "equity": float(equity), "source": source}]),
        keys=["date"])


def equity_series(store) -> pd.Series:
    e = store.read_table("equity")
    if e.empty:
        return pd.Series(dtype=float)
    s = pd.Series(e["equity"].astype(float).values, index=pd.to_datetime(e["date"])).sort_index()
    return s[~s.index.duplicated(keep="last")]


def drawdown_state(eq: pd.Series, asof) -> tuple[float, float]:
    """(month-to-date %, drawdown from peak %)."""
    if eq.empty:
        return 0.0, 0.0
    asof = pd.Timestamp(asof)
    eq = eq[eq.index <= asof]
    if eq.empty:
        return 0.0, 0.0
    month_start = eq[eq.index < asof.replace(day=1)]
    ref = month_start.iloc[-1] if len(month_start) else eq.iloc[0]
    mtd = 100 * (eq.iloc[-1] / ref - 1)
    dd = 100 * (eq.iloc[-1] / eq.cummax().iloc[-1] - 1)
    return float(mtd), float(dd)


def metrics(eq: pd.Series, trades: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    out = {}
    if len(eq) > 20:
        r = eq.pct_change().dropna()
        out["sharpe_annualized"] = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
        out["return_annualized_pct"] = float(((eq.iloc[-1] / eq.iloc[0]) ** (252 / len(r)) - 1) * 100)
        out["volatility_annualized_pct"] = float(r.std() * np.sqrt(252) * 100)
        out["max_drawdown_pct"] = float(-100 * (eq / eq.cummax() - 1).min())
        if benchmark is not None and len(benchmark):
            b = benchmark.reindex(eq.index).ffill().pct_change()
            j = pd.concat([r, b], axis=1).dropna()
            out["corr_sp500"] = float(j.corr().iloc[0, 1]) if len(j) > 20 else np.nan
    closed = trades[trades["status"] == "CLOSED"] if not trades.empty else trades
    if not closed.empty:
        R = closed["r_multiple"].astype(float)
        out["n_trades_closed"] = int(len(R))
        out["win_rate_pct"] = float((R > 0).mean() * 100)
        out["avg_win_R"] = float(R[R > 0].mean()) if (R > 0).any() else np.nan
        out["expectancy_R"] = float(R.mean())
        out["total_pnl_usd"] = float(closed["realized_pnl_usd"].astype(float).sum())
    rows = []
    for k, v in out.items():
        gate = ""
        if k == "sharpe_annualized":
            gate = f">= {GATE_B['sharpe_min']}" + (" OK" if v >= GATE_B["sharpe_min"] else " NOT YET")
        elif k == "max_drawdown_pct":
            gate = f"<= {GATE_B['max_dd_max_pct']}" + (" OK" if v <= GATE_B["max_dd_max_pct"] else " FAIL")
        elif k == "expectancy_R":
            gate = f">= +{GATE_B['expectancy_R_min']} R" + (" OK" if v >= GATE_B["expectancy_R_min"] else " NOT YET")
        elif k == "corr_sp500":
            gate = f"|corr| < {GATE_B['corr_abs_max']}" + (" OK" if abs(v) < GATE_B["corr_abs_max"] else " FAIL")
        rows.append({"indicator": k, "value": v, "gate_B": gate})
    return pd.DataFrame(rows)
