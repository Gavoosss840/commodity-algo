"""Grain fair value: two stages (study section 4).

Stage 1  weather -> US yield deviation from trend (%)          [nowcast, our edge]
Stage 2  log(stocks/use), dollar, real rate, ... -> log real price  [fair value]

The yield nowcast changes the projected ending stocks (with demand rationing),
hence stocks/use, hence the fair value. Comparing fair value with the market
price gives z. The same nowcast versus USDA's figure gives the expected move on
the next WASDE (event model, used for the option-vs-future decision).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.prices import load_bars
from . import weather_features as wf
from .wls import WLSResult, daily_lambda, fit_wls, stability_alerts

DEFAULT_NOWCAST_WEIGHT = {6: 1.0, 7: 1.0, 8: 0.6, 9: 0.4, 10: 0.2}


@dataclass
class MarketAnalysis:
    market: str
    asof: pd.Timestamp
    status: str                       # OK / NO_DATA / SUSPENDED / MONITOR
    price: float = np.nan
    fair_value: float = np.nan
    z: float = np.nan
    gap_pct: float = np.nan           # ln(FV / P)
    resid_sd: float = np.nan          # model noise, log units
    expected_report_move_pct: float = np.nan
    model: WLSResult | None = None
    factors_today: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)
    nowcast: dict = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    bars: pd.DataFrame | None = None


def marketing_year(d: pd.Timestamp) -> int:
    """Reference crop year for prices on date d: new crop from May (first WASDE projection)."""
    return d.year if d.month >= 5 else d.year - 1


def _by_my(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Map an annual series (period = MY-05-01) onto daily dates by marketing year."""
    if series.empty:
        return pd.Series(np.nan, index=dates)
    lut = {p.year: v for p, v in series.items()}
    return pd.Series([lut.get(marketing_year(d), np.nan) for d in dates], index=dates)


def _ffill_to(s: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 0) -> pd.Series:
    if s.empty:
        return pd.Series(np.nan, index=dates)
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s.reindex(s.index.union(dates)).sort_index().ffill().reindex(dates)


# ----------------------------------------------------------------------------
# Balance sheet
# ----------------------------------------------------------------------------
def current_balance(store, market: str, asof) -> dict:
    """USDA balance sheet for the reference marketing year, as known on asof.
    Manual WASDE rows (input/wasde_balance.csv) override PSD when more recent."""
    asof = pd.Timestamp(asof)
    my = marketing_year(asof)
    period = pd.Timestamp(my, 5, 1)
    bal = {"marketing_year": my, "source": None}
    for name in ("area", "yield", "beginning_stocks", "production", "imports",
                 "total_use", "ending_stocks", "stocks_to_use"):
        s = store.get_series(f"{market}_psd_{name}", as_of=asof)
        bal[name] = float(s.get(period, np.nan)) if len(s) else np.nan
    if not np.isnan(bal["ending_stocks"]):
        bal["source"] = "USDA PSD"
    # manual WASDE override (latest report as-of)
    es = store.get_series(f"{market}_ending_stocks", as_of=asof)
    if len(es):
        rd = es.index.max()
        if marketing_year(rd) == my:
            manual = {k: store.get_series(f"{market}_{c}", as_of=asof).get(rd, np.nan) for k, c in [
                ("area", "area_harvested_macres"), ("yield", "usda_yield"),
                ("beginning_stocks", "beginning_stocks"), ("production", "production"),
                ("imports", "imports"), ("total_use", "total_use"), ("ending_stocks", "ending_stocks")]}
            for k, v in manual.items():
                if not pd.isna(v):
                    bal[k] = float(v)
            bal["source"] = f"WASDE {rd.date()} (manual)"
    if bal.get("total_use") and not np.isnan(bal["total_use"]) and not np.isnan(bal["ending_stocks"]):
        bal["stocks_to_use"] = bal["ending_stocks"] / bal["total_use"]
    return bal


def trend_yield(store, market: str, my: int, mcfg: dict) -> float:
    """Linear trend of the last 30 years of USDA yields (PSD); config fallback."""
    y = store.get_series(f"{market}_psd_yield")
    y = y[(y.index.year < my) & (y.index.year >= my - 30)].dropna()
    if len(y) >= 15:
        coef = np.polyfit(y.index.year.astype(float), y.values, 1)
        return float(np.polyval(coef, my))
    t = mcfg.get("trend_yield", {})
    if t:
        return float(t["base"] + t.get("slope", 0.0) * (my - t["base_year"]))
    return np.nan


def yield_nowcast(store, market: str, mcfg: dict, asof, bal: dict) -> dict:
    betas = mcfg.get("yield_factors") or {}
    out = {"weather_weight": 0.0, "yield_dev_pct": 0.0, "features": {}, "share_known": {}}
    if not betas or not mcfg.get("weather_points"):
        return out
    asof = pd.Timestamp(asof)
    obs, fc = wf.load_daily_weather(store, mcfg["weather_points"], asof)
    if obs.empty:
        out["warning"] = "no weather data (run update)"
        return out
    year = asof.year
    feats, share = wf.features_for_year(obs, fc, year, asof, tuple(mcfg.get("weather_season_months", [6, 7, 8, 9])))
    dev, contrib = wf.yield_deviation_pct(feats, betas)
    weights = {int(k): v for k, v in (mcfg.get("nowcast_weight_by_month") or DEFAULT_NOWCAST_WEIGHT).items()}
    w = weights.get(asof.month, 0.0) if marketing_year(asof) == year else 0.0
    ty = trend_yield(store, market, year, mcfg)
    wy = ty * (1 + dev / 100.0) if not np.isnan(ty) else np.nan
    usda_y = bal.get("yield", np.nan)
    blended = usda_y if (w == 0 or np.isnan(wy)) else (w * wy + (1 - w) * (usda_y if not np.isnan(usda_y) else wy))
    out.update({"weather_weight": w, "yield_dev_pct": dev, "features": feats, "contrib": contrib,
                "share_known": share, "trend_yield": ty, "weather_yield": wy, "usda_yield": usda_y,
                "our_yield": blended})
    return out


def stocks_to_use_nowcast(bal: dict, our_yield: float, beta_su: float, rationing: float) -> dict:
    """Ending stocks after our yield, with demand rationing solved by fixed point:
    use reacts to the price change implied by the stocks/use change."""
    res = {"su_usda": bal.get("stocks_to_use", np.nan)}
    if any(np.isnan(bal.get(k, np.nan)) for k in ("area", "ending_stocks", "total_use")) or np.isnan(our_yield):
        res.update(su_nowcast=res["su_usda"], d_production=0.0, ending_stocks_nowcast=bal.get("ending_stocks", np.nan))
        return res
    d_prod = bal["area"] * (our_yield - bal["yield"])
    use0, es0 = bal["total_use"], bal["ending_stocks"]
    use = use0
    for _ in range(50):
        es = es0 + d_prod + (use0 - use)
        es = max(es, 0.05 * use)  # pipeline minimum
        dlnp = beta_su * (np.log(es / use) - np.log(es0 / use0))
        new_use = use0 * (1 + rationing * dlnp)
        if abs(new_use - use) < 1e-6:
            break
        use = new_use
    res.update(su_nowcast=es / use, d_production=d_prod, ending_stocks_nowcast=es, use_nowcast=use)
    return res


# ----------------------------------------------------------------------------
# Stage 2: fair value
# ----------------------------------------------------------------------------
def build_factor_frame(store, market: str, mcfg: dict, dates: pd.DatetimeIndex, asof) -> pd.DataFrame:
    cols = {}
    for fname, spec in mcfg["price_factors"].items():
        sid = spec["series"]
        if sid.endswith("_stocks_to_use_nowcast"):
            s = _by_my(store.get_series(f"{market}_psd_stocks_to_use", as_of=asof), dates)
        elif sid in ("corn_china_imports", "soybeans_brazil_production", "wheat_world_ex_us_china_su"):
            s = _by_my(store.get_series(sid, as_of=asof), dates)
        else:
            s = _ffill_to(store.get_series(sid, as_of=asof), dates)
        if spec.get("transform") == "log":
            s = np.log(s.where(s > 0))
        cols[fname] = s
    return pd.DataFrame(cols, index=dates)


def analyze_grain(store, settings, market: str, asof) -> MarketAnalysis:
    mcfg = settings.markets[market]
    mset = settings.model
    asof = pd.Timestamp(asof).normalize()
    an = MarketAnalysis(market=market, asof=asof, status="OK")

    bars = load_bars(store, market, as_of=asof)
    if bars.empty:
        an.status = "NO_DATA"
        an.alerts.append("no price history: run `commodity-algo update` with IB running (or Yahoo fallback)")
        return an
    an.bars = bars
    dates = bars.index
    an.price = float(bars["close"].iloc[-1])

    # real price
    cpi = _ffill_to(store.get_series(mcfg.get("deflator", "CPIAUCSL"), as_of=asof), dates, lag_days=45)
    if cpi.isna().all():
        cpi = pd.Series(1.0, index=dates)
        an.alerts.append("CPI missing: nominal prices used")
    cpi = cpi.bfill()
    y = np.log(bars["close"] / cpi)

    X = build_factor_frame(store, market, mcfg, dates, asof)
    usable = [c for c in X.columns if X[c].notna().sum() > 20]
    for c in X.columns.difference(usable):
        an.alerts.append(f"factor {c} has no data: absorbed into intercept")
    X = X[usable]
    # hold the last known value for annual/stepwise factors on today's row
    X = X.ffill()

    prior = pd.Series({k: v["beta"] for k, v in mcfg["price_factors"].items()})
    prior_se = pd.Series({k: v.get("se", np.nan) for k, v in mcfg["price_factors"].items()})
    lam = daily_lambda(mset["forgetting_lambda_weekly"])
    full = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(full) < 60:
        an.status = "NO_DATA"
        an.alerts.append(f"only {len(full)} complete daily rows (need 60): backfill FRED/PSD/prices")
        return an
    if len(full) >= mset["min_obs_reestimate"] and all(full[c].std() > 0 for c in usable):
        res = fit_wls(full["y"], full[usable], lam=lam, prior=prior,
                      prior_strength_obs=mset["prior_strength_obs"])
    else:
        win = full.iloc[-mset["calibration_window_days"]:]
        res = fit_wls(win["y"], win[usable], lam=lam, prior=prior, fix_betas=True)
    res.alerts = stability_alerts(res, prior, prior_se, mset["beta_alert_sigma"], mset["residual_alert_ratio"])
    an.model = res
    an.alerts += res.alerts

    # today's factors with OUR stocks/use nowcast
    bal = current_balance(store, market, asof)
    yn = yield_nowcast(store, market, mcfg, asof, bal)
    beta_su = float(res.beta.get("log_stocks_to_use", prior.get("log_stocks_to_use", 0.0)))
    su = stocks_to_use_nowcast(bal, yn.get("our_yield", bal.get("yield", np.nan)), beta_su,
                               mcfg.get("demand_rationing_elasticity", 0.0))
    x_today = X.iloc[-1].copy()
    if "log_stocks_to_use" in x_today and not np.isnan(su.get("su_nowcast", np.nan)):
        x_today["log_stocks_to_use"] = np.log(su["su_nowcast"])
    an.factors_today = x_today.to_dict()
    an.contributions = res.contributions(x_today).to_dict()
    an.nowcast = {"balance": bal, "yield": yn, "stocks_to_use": su}
    if not np.isnan(su.get("su_nowcast", np.nan)) and not np.isnan(su.get("su_usda", np.nan)):
        an.expected_report_move_pct = 100 * beta_su * (np.log(su["su_nowcast"]) - np.log(su["su_usda"]))

    ln_fv_real = res.predict(x_today)
    an.fair_value = float(np.exp(ln_fv_real) * cpi.iloc[-1])
    an.resid_sd = res.resid_sd
    an.gap_pct = float(np.log(an.fair_value / an.price))
    an.z = float((np.log(an.price) - np.log(an.fair_value)) / res.resid_sd) if res.resid_sd > 0 else np.nan
    if any(a.startswith("RESIDUALS WIDENING") for a in an.alerts):
        an.status = "SUSPENDED"
    if bal.get("source") is None:
        an.alerts.append("no USDA balance sheet: set USDA_FAS_API_KEY or fill input/wasde_balance.csv")
    return an
