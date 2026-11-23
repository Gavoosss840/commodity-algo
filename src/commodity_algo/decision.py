"""Decision rules (strategy sections 1.3, 1.4 and 1.6).

Entry   |z| >= 2, catalyst within 15 business days, trend & carry filters not strongly against
Size    risk 0.5-1 % of capital / risk per contract (max of 2.5 ATR stop and report-day jump)
Stops   thesis (fair value moves against us), catastrophe (2.5 ATR, placed in IB), time (2 catalysts)
Target  dynamic: 1/2 of the gap closed -> exit part, 2/3 -> exit part, rest trails at 1 ATR
Instrument  option spread if our expected move / implied move >= 1.3 (or livestock), else futures
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data.inputs import business_days_between


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def atr(bars: pd.DataFrame, n: int = 20) -> float:
    h, l, c = bars["high"], bars["low"], bars["close"]
    if h.isna().all() or l.isna().all():
        return float(c.diff().abs().tail(n).mean() * 1.4)  # close-to-close proxy
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def trend_filter(bars: pd.DataFrame, direction: int, lookback: int, max_sigma: float) -> tuple[bool, str]:
    c = bars["close"].dropna()
    if len(c) < lookback + 5 or direction == 0:
        return True, "n/a"
    r = math.log(c.iloc[-1] / c.iloc[-lookback - 1])
    sig = float(np.log(c).diff().tail(lookback).std() * math.sqrt(lookback))
    against = np.sign(r) == -direction
    strong = sig > 0 and abs(r) > max_sigma * sig
    txt = f"3m move {100*r:+.1f}% ({r/sig if sig else 0:+.1f} sigma)"
    return (not (against and strong)), txt


def carry_filter(bars: pd.DataFrame, direction: int, max_adverse_pct: float,
                 days_between_contracts: float = 75.0) -> tuple[bool, str]:
    if "next_close" not in bars or bars["next_close"].isna().iloc[-1] or direction == 0:
        return True, "n/a (no 2nd contract)"
    f, n = bars["close"].iloc[-1], bars["next_close"].iloc[-1]
    roll = math.log(f / n) * 365.0 / days_between_contracts * 100  # >0 backwardation: longs earn
    adverse = -roll * direction
    return adverse <= max_adverse_pct, f"roll yield {roll:+.1f}%/yr ({'backwardation' if roll > 0 else 'contango'})"


# ----------------------------------------------------------------------------
# Portfolio state
# ----------------------------------------------------------------------------
@dataclass
class AccountState:
    capital: float
    month_to_date_pct: float = 0.0
    drawdown_pct: float = 0.0
    open_risk_usd: float = 0.0
    open_risk_by_group: dict | None = None
    implied_vol: dict | None = None  # market -> IV (annualised, decimal)

    def size_multiplier(self, risk_cfg: dict) -> tuple[float, str]:
        if self.drawdown_pct <= risk_cfg["max_drawdown_stop_pct"]:
            return 0.0, f"drawdown {self.drawdown_pct:.1f}% <= {risk_cfg['max_drawdown_stop_pct']}%: TRADING STOPPED, review model"
        if self.month_to_date_pct <= risk_cfg["monthly_loss_halving_pct"]:
            return 0.5, f"month {self.month_to_date_pct:.1f}% <= {risk_cfg['monthly_loss_halving_pct']}%: size halved"
        return 1.0, "normal"


@dataclass
class Decision:
    market: str
    label: str
    asof: str
    status: str
    price: float
    fair_value: float
    gap_pct: float
    z: float
    model_noise_pct: float
    signal: str
    reason: str
    next_catalyst: str
    catalyst_bdays: float
    trend_check: str
    carry_check: str
    instrument: str
    contract: str
    quantity: int
    entry_limit: float
    catastrophe_stop: float
    target_1: float
    target_2: float
    trailing_stop_atr: float
    thesis_stop_fair_value: float
    time_stop: str
    risk_usd: float
    risk_pct: float
    atr20: float
    expected_move_pct: float
    implied_move_pct: float
    option_idea: str
    model_mode: str
    alerts: str
    suggestion_id: str

    def to_dict(self):
        return asdict(self)


def _round_tick(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 6) if tick else x


def decide(an, settings, state: AccountState, calendar: pd.DataFrame) -> Decision:
    mcfg = settings.markets[an.market]
    sig_cfg, risk_cfg = settings.signal, settings.risk
    asof = pd.Timestamp(an.asof)
    contracts = mcfg.get("contracts", {})
    micro = contracts.get("micro")
    std = contracts.get("standard")
    tick = (micro or std or {}).get("tick", 0.01)

    base = dict(market=an.market, label=mcfg.get("label", an.market), asof=asof.strftime("%Y-%m-%d"),
                status=an.status, price=an.price, fair_value=an.fair_value,
                gap_pct=100 * an.gap_pct if not np.isnan(an.gap_pct) else np.nan, z=an.z,
                model_noise_pct=100 * an.resid_sd if not np.isnan(an.resid_sd) else np.nan,
                signal="NONE", reason="", next_catalyst="", catalyst_bdays=np.nan,
                trend_check="", carry_check="", instrument="", contract="", quantity=0,
                entry_limit=np.nan, catastrophe_stop=np.nan, target_1=np.nan, target_2=np.nan,
                trailing_stop_atr=np.nan, thesis_stop_fair_value=np.nan, time_stop="",
                risk_usd=0.0, risk_pct=0.0, atr20=np.nan,
                expected_move_pct=an.expected_report_move_pct, implied_move_pct=np.nan,
                option_idea="", model_mode=an.model.mode if an.model else "",
                alerts=" | ".join(an.alerts),
                suggestion_id=f"{an.market}-{asof:%Y%m%d}")

    if an.status in ("NO_DATA", "SUSPENDED") or np.isnan(an.z):
        base["reason"] = f"model status {an.status}"
        return Decision(**base)

    a20 = atr(an.bars, sig_cfg["atr_period"]) if an.bars is not None else np.nan
    base["atr20"] = a20

    # catalyst
    events = mcfg.get("catalysts", [])
    cal = calendar[calendar["event"].isin(events) & calendar["markets"].fillna("").str.contains(an.market)
                   & (calendar["date"] > asof)] if not calendar.empty else calendar
    if cal is not None and len(cal):
        nxt = cal.iloc[0]
        base["next_catalyst"] = f"{nxt['event']} {nxt['date']:%Y-%m-%d}"
        base["catalyst_bdays"] = business_days_between(asof, nxt["date"])
        major = cal[cal["event"].isin(mcfg.get("major_reports", []))]
        later = major.iloc[1]["date"] if len(major) > 1 else (major.iloc[0]["date"] if len(major) else nxt["date"])
        base["time_stop"] = f"exit if gap not reduced after {sig_cfg['time_stop_catalysts']} reports (by {later:%Y-%m-%d})"

    direction = -int(np.sign(an.z))  # z<0: price below fair value -> BUY
    ok_trend, base["trend_check"] = trend_filter(an.bars, direction, sig_cfg["trend_filter"]["lookback_days"],
                                                 sig_cfg["trend_filter"]["max_adverse_sigma"])
    ok_carry, base["carry_check"] = carry_filter(an.bars, direction,
                                                 sig_cfg["carry_filter"]["max_adverse_annualized_pct"])

    # ---- entry checks
    reasons = []
    if an.status == "MONITOR" or not mcfg.get("enabled"):
        reasons.append("market in monitor mode (roadmap)")
    if abs(an.z) < sig_cfg["z_entry"]:
        reasons.append(f"|z| {abs(an.z):.2f} < {sig_cfg['z_entry']}")
    if np.isnan(base["catalyst_bdays"]) or base["catalyst_bdays"] > sig_cfg["catalyst_window_business_days"]:
        reasons.append("no dated catalyst within 15 business days")
    if not ok_trend:
        reasons.append("trend filter strongly against")
    if not ok_carry:
        reasons.append("carry filter strongly against")
    if sig_cfg.get("no_new_position_day_before_major_report") and len(cal):
        major_next = cal[cal["event"].isin(mcfg.get("major_reports", []))]
        if len(major_next) and business_days_between(asof, major_next.iloc[0]["date"]) <= 1:
            reasons.append("day before a major report: no new position")
    mult, mult_txt = state.size_multiplier(risk_cfg)
    if mult == 0:
        reasons.append(mult_txt)

    # ---- levels (computed even without a signal, for information)
    p, fv = an.price, an.fair_value
    gap = fv - p
    stop_dist = sig_cfg["catastrophe_atr_mult"] * a20
    base.update(
        entry_limit=_round_tick(p, tick),
        catastrophe_stop=_round_tick(p - direction * stop_dist, tick),
        target_1=_round_tick(p + sig_cfg["partial_exit_1_fraction_of_gap"] * gap, tick),
        target_2=_round_tick(p + sig_cfg["partial_exit_2_fraction_of_gap"] * gap, tick),
        trailing_stop_atr=sig_cfg["trailing_atr_mult"] * a20,
        thesis_stop_fair_value=_round_tick(fv - sig_cfg["thesis_break_fraction"] * gap, tick),
    )

    if reasons:
        base["reason"] = "; ".join(reasons)
        return Decision(**base)

    # ---- sizing
    group = mcfg.get("group", "")
    capital = state.capital
    budget = capital * risk_cfg["risk_per_trade_pct"] / 100.0 * mult
    room_total = capital * risk_cfg["max_total_open_risk_pct"] / 100.0 - state.open_risk_usd
    room_group = capital * risk_cfg["max_group_risk_pct"] / 100.0 - (state.open_risk_by_group or {}).get(group, 0.0)
    budget = max(0.0, min(budget, room_total, room_group))
    jump = mcfg.get("jump_allowance", 0.0)
    risk_points = max(stop_dist, jump)

    choice, qty, risk_usd = "", 0, 0.0
    if std:
        q_std = int(budget // (risk_points * std["point_value_usd"]))
        if q_std >= 2:
            choice, qty, risk_usd = std["symbol"], q_std, q_std * risk_points * std["point_value_usd"]
    if not choice and micro:
        q_mi = int(budget // (risk_points * micro["point_value_usd"]))
        if q_mi >= 1:
            choice, qty, risk_usd = micro["symbol"], q_mi, q_mi * risk_points * micro["point_value_usd"]
    if not choice and std:
        q_std = int(budget // (risk_points * std["point_value_usd"]))
        if q_std >= 1:
            choice, qty, risk_usd = std["symbol"], q_std, q_std * risk_points * std["point_value_usd"]

    # ---- instrument: option spread vs futures
    iv = (state.implied_vol or {}).get(an.market)
    our_move = abs(an.gap_pct) * 100
    if not np.isnan(an.expected_report_move_pct):
        our_move = max(our_move, abs(an.expected_report_move_pct))
    instrument = "FUTURE"
    if iv:
        days = (base["catalyst_bdays"] if not np.isnan(base["catalyst_bdays"]) else 10) * 7 / 5 + 3
        implied = iv * math.sqrt(days / 365.0) * 100
        base["implied_move_pct"] = implied
        if implied > 0 and our_move / implied >= sig_cfg["option_ratio_min"]:
            instrument = "OPTION_SPREAD"
    if group == "livestock":
        instrument = "OPTION_SPREAD"
    side = "BUY" if direction > 0 else "SELL"
    if instrument == "OPTION_SPREAD":
        k1 = _round_tick(p, 5 if "cents" in mcfg.get("price_unit", "") else 1)
        k2 = _round_tick(p + 0.9 * gap, 5 if "cents" in mcfg.get("price_unit", "") else 1)
        kind = "call" if direction > 0 else "put"
        base["option_idea"] = (f"BUY {kind} spread {mcfg.get('options_symbol', '')} {k1:g}/{k2:g}, expiry just after "
                               f"{base['next_catalyst']}; max debit {budget:,.0f} USD; never sell naked options")
        base.update(instrument="OPTION_SPREAD", contract=mcfg.get("options_symbol", ""), quantity=0,
                    risk_usd=budget, risk_pct=100 * budget / capital)
    else:
        if qty == 0:
            base["reason"] = (f"risk per contract {risk_points:.2f} pts x {micro['point_value_usd'] if micro else std['point_value_usd']} "
                              f"USD exceeds budget {budget:,.0f} USD")
            return Decision(**base)
        base.update(instrument="MICRO_FUTURE" if micro and choice == micro["symbol"] else "FUTURE",
                    contract=choice, quantity=qty, risk_usd=risk_usd, risk_pct=100 * risk_usd / capital)

    base["signal"] = side
    base["reason"] = (f"|z|={abs(an.z):.2f}>= {sig_cfg['z_entry']}, catalyst in {base['catalyst_bdays']:.0f} bd; "
                      f"sizing {mult_txt}; risk/contract uses max(2.5 ATR, report jump {jump})")
    return Decision(**base)
