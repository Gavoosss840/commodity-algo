"""Daily run: update data -> ingest your Excel inputs -> models -> decisions -> Excel.

    raw data (FRED, USDA, NOAA/Open-Meteo, EIA, IB)  --> bitemporal store
    store --> fair value + nowcast per market --> z --> rules --> fiche (Excel)
    fiche (your fills) --> journal --> performance --> recalibration
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import excel_io, journal, performance
from .config import Settings
from .data import eia, fred, inputs, prices, psd, weather
from .decision import AccountState, decide
from .models.grains import analyze_grain
from .models.oil import analyze_oil
from .storage import Store

log = logging.getLogger("commodity_algo")

FRED_SERIES = ["DTWEXBGS", "DFII10", "CPIAUCSL", "VIXCLS", "DHHNGSP", "DCOILWTICO", "PCOPPUSDM", "SP500"]
MODELS = {"grain_two_stage": analyze_grain, "oil_weekly_event": analyze_oil}

README_LINES = [
    "This fiche only PROPOSES trades. You place the orders yourself in Interactive Brokers.",
    "Decision sheet: one row per market. signal = BUY / SELL / NONE. 'reason' says why.",
    "z < 0: market below fair value (buy). z > 0: above (sell). Entry needs |z| >= 2 + a catalyst within 15 business days.",
    "Place the catastrophe_stop in IB at entry. target_1 / target_2: partial exits. thesis_stop_fair_value: exit if the "
    "model's fair value crosses this level (shown daily in Positions).",
    "AFTER YOU TRADE: in the YELLOW columns write EXECUTED=Y, FILL_PRICE, QTY_FILLED (and a THESIS_NOTE), then save.",
    "AFTER AN EXIT: in Positions write EXIT_QTY and EXIT_PRICE on the trade's row, then save.",
    "The next run reads these yellow cells once, records the trade in the journal, and tracks P&L in R multiples.",
    "Never edit other columns: they are regenerated. The full history lives in output/history/master_history.xlsx.",
    "Grain prices are in cents per bushel (450 = 4.50 USD). MZC = micro corn: 1 cent = 5 USD; ZC: 1 cent = 50 USD.",
]


def full_calendar(settings: Settings, asof) -> pd.DataFrame:
    cal = inputs.read_calendar(settings.path("calendar_input"))
    asof = pd.Timestamp(asof).normalize()
    days = pd.date_range(asof - pd.Timedelta(days=120), asof + pd.Timedelta(days=120))
    auto = []
    for d in days:
        if d.weekday() == 2:
            auto.append({"date": d, "event": "EIA_WPSR", "markets": "wti", "time_et": "10:30", "verified": "auto"})
        if d.weekday() == 0 and 4 <= d.month <= 11:
            auto.append({"date": d, "event": "CROP_PROGRESS", "markets": "corn|soybeans|wheat",
                         "time_et": "16:00", "verified": "auto"})
        if d.weekday() == 3:
            auto.append({"date": d, "event": "EXPORT_SALES", "markets": "corn|soybeans|wheat|lean_hogs|live_cattle",
                         "time_et": "08:30", "verified": "auto"})
    out = pd.concat([cal, pd.DataFrame(auto)], ignore_index=True) if auto else cal
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------
def update_data(store: Store, settings: Settings, ib=None, full: bool = False) -> list[str]:
    msgs = []

    def step(name, fn):
        try:
            r = fn()
            msgs.append(f"OK   {name}: {r if not isinstance(r, dict) else {k: v for k, v in r.items() if k != 'contracts'}}")
            return r
        except Exception as e:
            msgs.append(f"FAIL {name}: {type(e).__name__}: {e}")
            log.debug(traceback.format_exc())

    step("FRED macro", lambda: fred.update_fred(store, FRED_SERIES))
    step("WASDE manual input", lambda: inputs.store_wasde(store, inputs.read_wasde(settings.path("wasde_input"))))
    psd_key = settings.secret("USDA_FAS_API_KEY")
    contract_info = {}
    for m, mcfg in settings.enabled_markets().items():
        if mcfg.get("model") == "grain_two_stage":
            step(f"USDA PSD {m}", lambda m=m: psd.update_psd(store, m, psd_key, full=full))
            if mcfg.get("weather_points"):
                step(f"weather {m}", lambda mcfg=mcfg: weather.update_weather(store, mcfg["weather_points"]))
        if mcfg.get("model") == "oil_weekly_event":
            step("EIA weekly", lambda: eia.update_eia(store, settings.secret("EIA_API_KEY")))
        if ib is not None:
            r = step(f"IB prices {m}", lambda m=m, mcfg=mcfg: prices.update_prices_ib(store, ib, m, mcfg))
            if r:
                contract_info[m] = r.get("contracts", {})
        else:
            step(f"Yahoo prices {m} (fallback, IB not connected)",
                 lambda m=m, mcfg=mcfg: prices.update_prices_yahoo(store, m, mcfg))
    store._contract_info = contract_info
    return msgs


def connect_ib(settings: Settings, readonly: bool = True):
    try:
        from .data.ib_client import IBClient
        c = IBClient(settings.ib["host"], settings.ib["port"], settings.ib["client_id"],
                     settings.ib["market_data_type"], readonly=readonly)
        return c.connect()
    except Exception as e:
        log.warning("IB not connected (%s). Is TWS / IB Gateway running with the API enabled?", e)
        return None


# ----------------------------------------------------------------------------
def _model_rows(analyses: dict, settings: Settings) -> pd.DataFrame:
    rows = []
    for m, an in analyses.items():
        mcfg = settings.markets[m]
        study = mcfg.get("price_factors") or mcfg.get("event_factors") or {}
        if an.model is not None:
            for f, b in an.model.beta.items():
                rows.append({"market": m, "section": "beta", "item": f, "beta_used": b,
                             "beta_study": study.get(f, {}).get("beta"), "se_now": an.model.se.get(f),
                             "value_today": an.factors_today.get(f), "contribution_log": an.contributions.get(f),
                             "detail": f"mode={an.model.mode}, n={an.model.n_obs}, R2={an.model.r2:.2f}, "
                                       f"noise={100*an.model.resid_sd:.1f}%"})
        nc = an.nowcast or {}
        for k, v in (nc.get("balance") or {}).items():
            rows.append({"market": m, "section": "USDA balance", "item": k, "value_today": v if not isinstance(v, str) else None,
                         "detail": v if isinstance(v, str) else ""})
        y = nc.get("yield") or {}
        for k in ("trend_yield", "weather_yield", "usda_yield", "our_yield", "yield_dev_pct", "weather_weight"):
            if k in y:
                rows.append({"market": m, "section": "yield nowcast", "item": k, "value_today": y[k]})
        for k, v in (y.get("features") or {}).items():
            rows.append({"market": m, "section": "weather feature", "item": k, "value_today": v,
                         "contribution_log": (y.get("contrib") or {}).get(k)})
        for k, v in (y.get("share_known") or {}).items():
            rows.append({"market": m, "section": "weather share observed/forecast", "item": k, "value_today": v})
        for k, v in (nc.get("stocks_to_use") or {}).items():
            rows.append({"market": m, "section": "stocks/use nowcast", "item": k, "value_today": v})
    return pd.DataFrame(rows)


def _risk_rows(settings, state: AccountState, mult_txt: str, acct: dict | None) -> pd.DataFrame:
    r = settings.risk
    cap = state.capital
    rows = [
        {"item": "capital (USD)", "value": cap, "comment": "IB NetLiquidation" if acct else "config account.capital_usd (IB not connected)"},
        {"item": "risk budget per trade (USD)", "value": cap * r["risk_per_trade_pct"] / 100, "comment": f"{r['risk_per_trade_pct']}% of capital"},
        {"item": "open risk total (USD)", "value": state.open_risk_usd,
         "comment": f"{100*state.open_risk_usd/cap:.2f}% used of max {r['max_total_open_risk_pct']}%"},
    ]
    for g, v in (state.open_risk_by_group or {}).items():
        rows.append({"item": f"open risk {g} (USD)", "value": v, "comment": f"{100*v/cap:.2f}% of max {r['max_group_risk_pct']}%"})
    rows += [
        {"item": "month-to-date P&L (%)", "value": state.month_to_date_pct, "comment": f"size halved at {r['monthly_loss_halving_pct']}%"},
        {"item": "drawdown from peak (%)", "value": state.drawdown_pct, "comment": f"full stop at {r['max_drawdown_stop_pct']}%"},
        {"item": "size multiplier", "value": state.size_multiplier(r)[0], "comment": mult_txt},
    ]
    if acct:
        cash_ok = acct["cash"] >= r["cash_margin_multiple"] * acct["init_margin"] if acct["init_margin"] == acct["init_margin"] else True
        rows.append({"item": "cash vs initial margin", "value": f"{acct['cash']:,.0f} / {acct['init_margin']:,.0f}",
                     "comment": "OK" if cash_ok else f"BELOW {r['cash_margin_multiple']}x initial margin"})
        rows.append({"item": "account", "value": acct["account"], "comment": "PAPER" if acct["is_paper"] else "LIVE"})
    return pd.DataFrame(rows)


def _preserve_inputs(path: Path, decisions: pd.DataFrame, positions: pd.DataFrame):
    """If today's fiche already exists (re-run), keep what you typed in the yellow cells."""
    if not path.exists():
        return decisions, positions
    try:
        old_d = pd.read_excel(path, sheet_name="Decision")
        old_p = pd.read_excel(path, sheet_name="Positions")
    except Exception:
        return decisions, positions
    if "suggestion_id" in old_d and not decisions.empty:
        keep = [c for c in excel_io.DECISION_INPUTS if c in old_d]
        decisions = decisions.merge(old_d[["suggestion_id"] + keep], on="suggestion_id", how="left")
    if "trade_id" in old_p and not positions.empty:
        keep = [c for c in excel_io.POSITION_INPUTS if c in old_p]
        positions = positions.drop(columns=["exit_qty", "exit_price"], errors="ignore").merge(
            old_p[["trade_id"] + keep], on="trade_id", how="left")
    return decisions, positions


# ----------------------------------------------------------------------------
def run_daily(settings: Settings, asof=None, use_ib: bool = True, update: bool = True,
              stage_orders: bool = False, full_backfill: bool = False) -> dict:
    asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
    store = Store(settings.path("database"))
    alerts: list[str] = []

    ib = connect_ib(settings, readonly=not stage_orders) if use_ib else None
    if use_ib and ib is None:
        alerts.append("Interactive Brokers not connected: prices from Yahoo fallback, capital from config")
    acct = None
    try:
        if update:
            alerts += [m for m in update_data(store, settings, ib, full=full_backfill) if m.startswith("FAIL")]
        if ib is not None:
            acct = ib.account()
            performance.record_equity(store, asof, acct["net_liquidation"], f"IB {acct['account']}")
            if not acct["is_paper"]:
                alerts.append("LIVE ACCOUNT connected")
            ib_pos = ib.positions()
            symbols = {c["symbol"] for mc in settings.markets.values() for c in mc.get("contracts", {}).values()}
            if not ib_pos.empty:
                ib_pos = ib_pos[ib_pos["sec_type"].isin(["FUT", "FOP"]) & ib_pos["symbol"].isin(symbols)]
        else:
            ib_pos = pd.DataFrame()

        # 1) Your Excel inputs -> journal
        alerts += [f"journal: {m}" for m in excel_io.ingest_user_inputs(store, settings, settings.path("daily_output_dir"))]

        # 2) Models
        calendar = full_calendar(settings, asof)
        analyses = {}
        for m, mcfg in settings.markets.items():
            fn = MODELS.get(mcfg.get("model"))
            if fn is None or (not mcfg.get("enabled") and not mcfg.get("monitor", False)):
                continue
            try:
                analyses[m] = fn(store, settings, m, asof)
            except Exception as e:
                alerts.append(f"{m}: model error {type(e).__name__}: {e}")
                log.debug(traceback.format_exc())
        for m, mcfg in settings.enabled_markets().items():
            if mcfg.get("model") == "grain_two_stage":
                alerts += [f"input: {p}" for p in inputs.validate_wasde(inputs.read_wasde(settings.path("wasde_input")), m)
                           if store.get_series(f"{m}_psd_stocks_to_use").empty]

        # 3) Open positions
        positions = journal.manage_positions(store, settings, analyses, calendar, asof)
        if not ib_pos.empty or not positions.empty:
            ib_q = ib_pos.groupby("local_symbol")["position"].sum() if not ib_pos.empty else pd.Series(dtype=float)
            j_q = (positions["qty_open"] * np.where(positions["direction"] == "LONG", 1, -1)).groupby(
                positions["contract"]).sum() if not positions.empty else pd.Series(dtype=float)
            if ib is not None:
                for c in ib_q.index.union(j_q.index):
                    q, jq = float(ib_q.get(c, 0.0)), float(j_q.get(c, 0.0))
                    if abs(q - jq) > 1e-9:
                        alerts.append(f"RECONCILE {c}: IB position {q:g} vs journal {jq:g} "
                                      f"(record the fill in the fiche, or check IB)")

        # 4) Account state
        eq = performance.equity_series(store)
        if acct is None:
            trades = store.read_table("trades")
            realized = float(trades["realized_pnl_usd"].astype(float).sum()) if not trades.empty else 0.0
            unreal = float(positions["unrealized_pnl_usd"].fillna(0).sum()) if not positions.empty else 0.0
            capital = settings.account["capital_usd"] + realized + unreal
            performance.record_equity(store, asof, capital, "config + journal")
            eq = performance.equity_series(store)
        else:
            capital = acct["net_liquidation"]
        mtd, dd = performance.drawdown_state(eq, asof)
        open_risk = float(positions["risk_open_usd"].fillna(0).sum()) if not positions.empty else 0.0
        by_group = positions.groupby("group")["risk_open_usd"].sum().to_dict() if not positions.empty else {}
        ivs = {}
        if ib is not None:
            for m, info in getattr(store, "_contract_info", {}).items():
                kind = info.get("standard") or info.get("micro")
                if kind:
                    ivs[m] = ib.implied_vol(kind["info"])
        state = AccountState(capital=capital, month_to_date_pct=mtd, drawdown_pct=dd, open_risk_usd=open_risk,
                             open_risk_by_group=by_group, implied_vol=ivs)
        mult, mult_txt = state.size_multiplier(settings.risk)

        # 5) Decisions
        decisions = pd.DataFrame([decide(an, settings, state, calendar).to_dict() for an in analyses.values()])
        if not decisions.empty:
            # micro contract symbol actually tradable in IB
            for i, r in decisions.iterrows():
                info = getattr(store, "_contract_info", {}).get(r["market"], {})
                for kind in ("micro", "standard"):
                    if kind in info and info[kind]["info"].symbol == r["contract"]:
                        decisions.at[i, "contract"] = info[kind]["info"].local_symbol
            for an in analyses.values():
                alerts += [f"{an.market}: {a}" for a in an.alerts]

        # 6) Optional: stage orders in TWS (NOT transmitted)
        if stage_orders and ib is not None and not decisions.empty:
            for _, r in decisions[decisions["signal"].isin(["BUY", "SELL"]) &
                                  decisions["instrument"].str.contains("FUTURE")].iterrows():
                info = getattr(store, "_contract_info", {}).get(r["market"], {})
                ci = next((v["info"] for v in info.values() if v["info"].local_symbol == r["contract"]), None)
                if ci is None:
                    continue
                try:
                    ib.stage_bracket(ci, r["signal"], int(r["quantity"]), r["entry_limit"], r["catastrophe_stop"],
                                     None, r["suggestion_id"], transmit=settings.ib["stage_orders_transmit"],
                                     allow_live=settings.ib["allow_live_account_orders"])
                    alerts.append(f"orders STAGED in TWS for {r['suggestion_id']} (not transmitted)")
                except Exception as e:
                    alerts.append(f"stage orders {r['suggestion_id']}: {e}")

        # 7) Persist: everything accumulates in the store
        run_date = asof.strftime("%Y-%m-%d")
        if not decisions.empty:
            store.upsert_table("signals", decisions.assign(run_date=run_date), keys=["run_date", "market"])
        if not positions.empty:
            store.upsert_table("positions_daily", positions.drop(columns=["exit_qty", "exit_price"]).assign(run_date=run_date),
                               keys=["run_date", "trade_id"])
        beta_rows = []
        for m, an in analyses.items():
            if an.model is not None:
                for f, b in an.model.beta.items():
                    beta_rows.append({"run_date": run_date, "market": m, "factor": f, "beta": b,
                                      "se": an.model.se.get(f), "mode": an.model.mode, "n_obs": an.model.n_obs,
                                      "r2": an.model.r2, "resid_sd": an.model.resid_sd,
                                      "intercept": an.model.intercept})
        store.upsert_table("betas", pd.DataFrame(beta_rows), keys=["run_date", "market", "factor"])

        # 8) Excel
        perf = performance.metrics(performance.equity_series(store), store.read_table("trades"),
                                   store.get_series("SP500"))
        daily_path = settings.path("daily_output_dir") / f"{asof:%Y}" / f"fiche_{run_date}.xlsx"
        decisions, positions = _preserve_inputs(daily_path, decisions, positions)
        fresh = store.data_freshness()
        cal_view = calendar[(calendar["date"] >= asof) & (calendar["date"] <= asof + pd.Timedelta(days=45))]
        excel_io.write_daily_fiche(daily_path, decisions, positions,
                                   _risk_rows(settings, state, mult_txt, acct), _model_rows(analyses, settings),
                                   fresh, cal_view, pd.DataFrame({"alert": alerts}), README_LINES)
        store.upsert_table("runs", pd.DataFrame([{
            "run_date": run_date, "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "ib_connected": ib is not None, "capital": capital, "n_signals": int(decisions["signal"].isin(["BUY", "SELL"]).sum()) if not decisions.empty else 0,
            "n_alerts": len(alerts), "fiche": str(daily_path.relative_to(settings.root)) if daily_path.is_relative_to(settings.root) else str(daily_path)}]),
            keys=["run_date"])
        hist_path = excel_io.write_history_workbook(store, settings.path("history_workbook"), perf)
        return {"fiche": daily_path, "history": hist_path, "decisions": decisions, "positions": positions,
                "alerts": alerts, "analyses": analyses}
    finally:
        if ib is not None:
            ib.disconnect()
