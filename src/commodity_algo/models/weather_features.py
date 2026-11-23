"""Weather -> yield features (study, section 4, etage 1).

Daily weather of the producing states is averaged, then aggregated by month:
  <mon>_tmax_anom_f        monthly mean Tmax minus its historical mean (deg F)
  <mon>_extreme_heat_f     degrees of monthly mean Tmax above its historical 75th pct
  <mon>_precip_anom_in     monthly rain minus its historical mean (inches)
  <mon>_precip_deficit_in  max(0, historical mean - monthly rain)
  <mon>_kdd29c             killing degree-days above 29 C (Schlenker-Roberts, diagnostic)

For the season in progress each day is taken, in order of preference, from
observations (ERA5), the latest 16-day forecast known on the as-of date, then
the climatology of that calendar day (last 30 years). `share_known` reports how
much of the month is observed or forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = {6: "jun", 7: "jul", 8: "aug", 9: "sep"}


def load_daily_weather(store, points: list[dict], asof) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (observed, forecast) daily frames averaged across points: tmax, prcp."""
    obs, fc = {}, {}
    for var in ("tmax", "prcp"):
        o = [store.get_series(f"wx_{p['name']}_{var}", as_of=asof) for p in points]
        f = [store.get_series(f"wxf_{p['name']}_{var}", as_of=asof) for p in points]
        o = [s for s in o if len(s)]
        f = [s for s in f if len(s)]
        obs[var] = pd.concat(o, axis=1).mean(axis=1) if o else pd.Series(dtype=float)
        fc[var] = pd.concat(f, axis=1).mean(axis=1) if f else pd.Series(dtype=float)
    return pd.DataFrame(obs), pd.DataFrame(fc)


def _climatology(obs: pd.DataFrame, year: int, nyears: int = 30) -> pd.DataFrame:
    hist = obs[(obs.index.year < year) & (obs.index.year >= year - nyears)]
    if hist.empty:
        return pd.DataFrame()
    return hist.groupby([hist.index.month, hist.index.day]).mean()


def _kdd(tmax_f: pd.Series, threshold_c: float = 29.0) -> float:
    tc = (tmax_f - 32.0) * 5.0 / 9.0
    return float(np.clip(tc - threshold_c, 0, None).sum())


def season_month(obs: pd.DataFrame, fc: pd.DataFrame, year: int, month: int, asof) -> tuple[pd.DataFrame, float]:
    """Daily tmax/prcp for one month, completed with forecast then climatology."""
    asof = pd.Timestamp(asof).normalize()
    days = pd.date_range(pd.Timestamp(year, month, 1), pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0))
    out = obs.reindex(days)
    known = out["tmax"].notna() if "tmax" in out else pd.Series(False, index=days)
    if not fc.empty:
        f = fc.reindex(days)
        fill = out["tmax"].isna() & f["tmax"].notna() & (days > asof - pd.Timedelta(days=7))
        out.loc[fill, :] = f.loc[fill, out.columns]
        known = known | fill
    clim = _climatology(obs, year)
    if out["tmax"].isna().any() and not clim.empty:
        for d in out.index[out["tmax"].isna()]:
            key = (d.month, d.day)
            if key in clim.index:
                out.loc[d] = clim.loc[key, out.columns].values
    return out, float(known.mean())


def monthly_history(obs: pd.DataFrame, last_full_year: int) -> pd.DataFrame:
    """Monthly mean tmax / total prcp / kdd per (year, month) for completed years."""
    o = obs[obs.index.year <= last_full_year]
    if o.empty:
        return pd.DataFrame()
    g = o.groupby([o.index.year, o.index.month])
    return pd.DataFrame({"tmax": g["tmax"].mean(), "prcp": g["prcp"].sum(),
                         "kdd29c": g["tmax"].apply(_kdd)}).rename_axis(["year", "month"])


def features_for_year(obs: pd.DataFrame, fc: pd.DataFrame, year: int, asof,
                      months=(6, 7, 8, 9)) -> tuple[dict, dict]:
    """Features for crop year `year` as known on `asof`. Returns (features, share_known)."""
    hist = monthly_history(obs, year - 1)
    feats, share = {}, {}
    for m in months:
        name = MONTHS[m]
        if hist.empty or m not in hist.index.get_level_values("month"):
            continue
        hm = hist.xs(m, level="month")
        days, sk = season_month(obs, fc, year, m, asof)
        if days["tmax"].isna().all():
            continue
        tmax, prcp = float(days["tmax"].mean()), float(days["prcp"].sum())
        feats[f"{name}_tmax_f"] = tmax
        feats[f"{name}_tmax_anom_f"] = tmax - hm["tmax"].mean()
        feats[f"{name}_extreme_heat_f"] = max(0.0, tmax - hm["tmax"].quantile(0.75))
        feats[f"{name}_precip_in"] = prcp
        feats[f"{name}_precip_anom_in"] = prcp - hm["prcp"].mean()
        feats[f"{name}_precip_deficit_in"] = max(0.0, hm["prcp"].mean() - prcp)
        feats[f"{name}_kdd29c"] = _kdd(days["tmax"])
        share[name] = sk
    return feats, share


def yield_deviation_pct(feats: dict, betas: dict) -> tuple[float, dict]:
    contrib = {k: betas[k] * feats.get(k, 0.0) for k in betas}
    missing = [k for k in betas if k not in feats]
    for k in missing:
        contrib[k] = 0.0
    return float(sum(contrib.values())), contrib
