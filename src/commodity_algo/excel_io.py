"""Excel outputs and the Excel -> algo feedback loop.

Daily fiche   output/daily/<YYYY>/fiche_<YYYY-MM-DD>.xlsx
  Decision   one row per market: signal, size, stops, targets + YELLOW input columns
  Positions  open trades with today's action + YELLOW exit columns
  Risk       portfolio risk usage, drawdown state, size multiplier
  Model      betas used, factor values, contributions, nowcast details
  Data       freshness of every stored series
  Calendar   upcoming catalysts
  Alerts     everything the algo wants you to look at

Master history  output/history/master_history.xlsx (rebuilt from the store each run)
  Signals_history, Positions_history, Trades, Fills, Equity, Performance,
  Betas_history, Runs

Feedback loop: after you trade, type in the yellow cells of the fiche and save.
The next run reads every fiche in output/daily, ingests what you typed (once),
and the trade enters the journal -> performance -> monthly recalibration.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import journal

DECISION_ORDER = [
    "market", "label", "signal", "instrument", "contract", "quantity", "entry_limit", "catastrophe_stop",
    "target_1", "target_2", "trailing_stop_atr", "thesis_stop_fair_value", "time_stop", "z", "price",
    "fair_value", "gap_pct", "model_noise_pct", "next_catalyst", "catalyst_bdays", "risk_usd", "risk_pct",
    "trend_check", "carry_check", "expected_move_pct", "implied_move_pct", "option_idea", "reason",
    "atr20", "model_mode", "status", "alerts", "suggestion_id", "asof",
]
DECISION_INPUTS = ["EXECUTED (Y/N)", "FILL_PRICE", "QTY_FILLED", "THESIS_NOTE"]
POSITION_INPUTS = ["EXIT_QTY", "EXIT_PRICE"]


def _write_sheet(writer, name: str, df: pd.DataFrame, input_cols=(), widths=None, signal_col=None):
    wb = writer.book
    df = df.copy() if df is not None else pd.DataFrame()
    if df.empty:
        df = pd.DataFrame({"info": ["(empty)"]})
    for c in input_cols:
        if c not in df:
            df[c] = None
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    head = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "text_wrap": True,
                          "valign": "top", "border": 1})
    head_in = wb.add_format({"bold": True, "bg_color": "#FFD966", "text_wrap": True, "valign": "top", "border": 1})
    inp = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    for j, c in enumerate(df.columns):
        ws.write(0, j, c, head_in if c in input_cols else head)
        w = (widths or {}).get(c, min(max(10, len(str(c)) + 2), 45))
        ws.set_column(j, j, w, inp if c in input_cols else None)
    ws.freeze_panes(1, 2)
    ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
    if signal_col and signal_col in df:
        j = list(df.columns).index(signal_col)
        green = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100", "bold": True})
        red = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True})
        ws.conditional_format(1, j, len(df), j, {"type": "cell", "criteria": "==", "value": '"BUY"', "format": green})
        ws.conditional_format(1, j, len(df), j, {"type": "cell", "criteria": "==", "value": '"SELL"', "format": red})


def write_daily_fiche(path: Path, decisions: pd.DataFrame, positions: pd.DataFrame, risk: pd.DataFrame,
                      model: pd.DataFrame, data: pd.DataFrame, calendar: pd.DataFrame, alerts: pd.DataFrame,
                      readme: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dec = decisions[[c for c in DECISION_ORDER if c in decisions]] if not decisions.empty else decisions
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        _write_sheet(w, "Decision", dec, DECISION_INPUTS, widths={"reason": 60, "alerts": 60, "option_idea": 50,
                                                                  "time_stop": 40, "label": 20}, signal_col="signal")
        _write_sheet(w, "Positions", positions, POSITION_INPUTS, widths={"action": 70, "trade_id": 30})
        _write_sheet(w, "Risk", risk, widths={"item": 40, "value": 25, "comment": 60})
        _write_sheet(w, "Model", model, widths={"detail": 60})
        _write_sheet(w, "Data", data, widths={"series_id": 34})
        _write_sheet(w, "Calendar", calendar)
        _write_sheet(w, "Alerts", alerts, widths={"alert": 110})
        pd.DataFrame({"How to use this fiche": readme}).to_excel(w, sheet_name="ReadMe", index=False)
        w.sheets["ReadMe"].set_column(0, 0, 120)
    return path


# ----------------------------------------------------------------------------
# Feedback loop: ingest what the user typed in the yellow cells
# ----------------------------------------------------------------------------
def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()


def _already(store, key: str) -> bool:
    t = store.read_table("ingested_inputs")
    return (not t.empty) and key in set(t["key"])


def _mark(store, key: str, file: str, what: str):
    store.upsert_table("ingested_inputs", pd.DataFrame([{
        "key": key, "file": file, "what": what, "ingested_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}]),
        keys=["key"])


def _yes(v) -> bool:
    return str(v).strip().upper() in ("Y", "YES", "O", "OUI", "1", "TRUE", "X")


def ingest_user_inputs(store, settings, daily_dir: Path) -> list[str]:
    log = []
    for f in sorted(Path(daily_dir).rglob("fiche_*.xlsx")):
        if f.name.startswith("~$"):
            continue
        try:
            dec = pd.read_excel(f, sheet_name="Decision")
            pos = pd.read_excel(f, sheet_name="Positions")
        except Exception as e:  # file open in Excel, corrupted...
            log.append(f"skip {f.name}: {e}")
            continue
        for _, r in dec.iterrows():
            if "EXECUTED (Y/N)" not in dec or not _yes(r.get("EXECUTED (Y/N)")):
                continue
            px, q = r.get("FILL_PRICE"), r.get("QTY_FILLED")
            if pd.isna(px) or pd.isna(q):
                log.append(f"{f.name} {r['market']}: EXECUTED=Y but FILL_PRICE/QTY_FILLED empty")
                continue
            k = _key(f.name, "open", r["suggestion_id"], px, q)
            if _already(store, k):
                continue
            direction = r["signal"] if r["signal"] in ("BUY", "SELL") else "BUY"
            tid = journal.open_trade(
                store, settings, market=r["market"], contract=str(r["contract"]), direction=direction,
                entry_date=r["asof"], entry_price=float(px), qty=int(q), fair_value=float(r["fair_value"]),
                z=float(r["z"]), catastrophe_stop=float(r["catastrophe_stop"]),
                thesis=str(r.get("THESIS_NOTE") or ""), suggestion_id=r["suggestion_id"],
                instrument=str(r["instrument"]))
            _mark(store, k, f.name, f"open {tid}")
            log.append(f"ingested entry {tid} from {f.name}")
        if "trade_id" in pos and "EXIT_QTY" in pos:
            for _, r in pos.iterrows():
                if pd.isna(r.get("EXIT_QTY")) or pd.isna(r.get("EXIT_PRICE")):
                    continue
                k = _key(f.name, "exit", r["trade_id"], r["EXIT_QTY"], r["EXIT_PRICE"])
                if _already(store, k):
                    continue
                date = f.stem.replace("fiche_", "")
                journal.close_part(store, r["trade_id"], date, int(r["EXIT_QTY"]), float(r["EXIT_PRICE"]), "excel")
                _mark(store, k, f.name, f"exit {r['trade_id']}")
                log.append(f"ingested exit {r['trade_id']} x{int(r['EXIT_QTY'])} from {f.name}")
    return log


def ingest_archive(store, daily_dir: Path) -> int:
    """Rebuild the signals history from the Excel archive (e.g. new machine / lost DB)."""
    n = 0
    for f in sorted(Path(daily_dir).rglob("fiche_*.xlsx")):
        if f.name.startswith("~$"):
            continue
        dec = pd.read_excel(f, sheet_name="Decision")
        if "suggestion_id" not in dec:
            continue
        dec = dec.drop(columns=[c for c in DECISION_INPUTS if c in dec])
        dec["run_date"] = f.stem.replace("fiche_", "")
        store.upsert_table("signals", dec, keys=["run_date", "market"])
        n += len(dec)
    return n


# ----------------------------------------------------------------------------
# Master history workbook
# ----------------------------------------------------------------------------
def write_history_workbook(store, path: Path, perf: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = {
        "Signals_history": store.read_table("signals"),
        "Positions_history": store.read_table("positions_daily"),
        "Trades": store.read_table("trades"),
        "Fills": store.read_table("fills"),
        "Equity": store.read_table("equity"),
        "Performance": perf,
        "Betas_history": store.read_table("betas"),
        "Runs": store.read_table("runs"),
    }
    for k in ("Signals_history", "Positions_history", "Betas_history", "Runs", "Equity"):
        df = sheets[k]
        sort = [c for c in ("run_date", "date", "market") if c in df]
        if not df.empty and sort:
            sheets[k] = df.sort_values(sort)
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        for name, df in sheets.items():
            _write_sheet(w, name, df, signal_col="signal" if name == "Signals_history" else None)
    return path
