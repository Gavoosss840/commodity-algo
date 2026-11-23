"""WTI weekly event model (study section 3) - MONITOR ONLY (roadmap stage 4).

Wednesday-to-Wednesday % change of WTI explained by VIX, dollar, and the
seasonally adjusted change in EIA stocks (change minus the average change of
the same week over the 5 previous years - the free proxy for the surprise).
Output: the move the factors imply this week versus the move observed. A large
gap (z) means price moved too far / not enough given the week's information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.prices import load_bars
from .grains import MarketAnalysis
from .wls import fit_wls


def _seasonal_change(s: pd.Series) -> pd.Series:
    d = s.diff()
    wk = d.index.isocalendar().week.values
    out = pd.Series(np.nan, index=d.index)
    for i, (t, v) in enumerate(d.items()):
        past = d[(wk == wk[i]) & (d.index < t) & (d.index >= t - pd.DateOffset(years=5, days=3))]
        if len(past) >= 3:
            out.iloc[i] = v - past.mean()
    return out


def analyze_oil(store, settings, market: str, asof) -> MarketAnalysis:
    mcfg = settings.markets[market]
    asof = pd.Timestamp(asof).normalize()
    an = MarketAnalysis(market=market, asof=asof, status="MONITOR")
    bars = load_bars(store, market, as_of=asof)
    if bars.empty:
        wti = store.get_series("DCOILWTICO", as_of=asof)
        if wti.empty:
            an.status = "NO_DATA"
            an.alerts.append("no WTI prices")
            return an
        bars = pd.DataFrame({"close": wti, "high": wti, "low": wti, "volume": np.nan})
    an.bars = bars
    weekly = bars["close"].resample("W-WED").last().dropna()
    y = 100 * weekly.pct_change()

    f = {}
    for name, spec in mcfg["event_factors"].items():
        s = store.get_series(spec["series"], as_of=asof)
        if s.empty:
            an.alerts.append(f"{name}: no data ({spec['series']})")
            continue
        if name.startswith("d_vix"):
            f[name] = s.resample("W-WED").last().diff()
        elif name.endswith("_pct"):
            f[name] = 100 * s.resample("W-WED").last().pct_change()
        elif name.endswith("_sa_change_mb"):
            f[name] = _seasonal_change(s.resample("W-WED").last())
        elif name.startswith("d_us_production"):
            f[name] = s.resample("W-WED").last().diff() / 100.0
    if not f:
        an.status = "NO_DATA"
        return an
    X = pd.DataFrame(f).reindex(y.index)
    prior = pd.Series({k: v["beta"] for k, v in mcfg["event_factors"].items()})
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 30:
        an.status = "NO_DATA"
        an.alerts.append(f"only {len(df)} complete weeks")
        return an
    res = fit_wls(df["y"].iloc[:-1], df[X.columns].iloc[:-1], prior=prior, fix_betas=True)
    an.model = res
    x_last = df[X.columns].iloc[-1]
    pred = res.predict(x_last) - res.intercept  # expected move from this week's information
    actual = df["y"].iloc[-1]
    an.price = float(bars["close"].iloc[-1])
    an.fair_value = float(weekly.loc[df.index[-1]] / (1 + actual / 100) * (1 + pred / 100))
    an.resid_sd = res.resid_sd / 100
    an.gap_pct = float(np.log(an.fair_value / an.price))
    an.z = float((actual - pred) / res.resid_sd) if res.resid_sd > 0 else np.nan
    an.expected_report_move_pct = float(pred)
    an.factors_today = x_last.to_dict()
    an.contributions = res.contributions(x_last).to_dict()
    return an
