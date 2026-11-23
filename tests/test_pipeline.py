"""End-to-end: daily run -> Excel fiche -> user types a fill -> next run journals it."""
import pandas as pd
from openpyxl import load_workbook

from commodity_algo import journal
from commodity_algo.backtest import BTConfig, MemoryStore, run_backtest
from commodity_algo.pipeline import run_daily


def test_daily_run_and_excel_feedback_loop(project):
    settings, store = project
    res = run_daily(settings, asof="2026-11-20", use_ib=False, update=False)
    fiche = res["fiche"]
    assert fiche.exists() and res["history"].exists()
    sheets = pd.ExcelFile(fiche).sheet_names
    for s in ("Decision", "Positions", "Risk", "Model", "Data", "Calendar", "Alerts", "ReadMe"):
        assert s in sheets
    dec = pd.read_excel(fiche, sheet_name="Decision")
    assert "corn" in set(dec["market"]) and "EXECUTED (Y/N)" in dec

    # --- the user trades and types the fill in the yellow cells
    wb = load_workbook(fiche)
    ws = wb["Decision"]
    head = [c.value for c in ws[1]]
    row = 2
    ws.cell(row, head.index("EXECUTED (Y/N)") + 1, "Y")
    ws.cell(row, head.index("FILL_PRICE") + 1, float(dec.loc[0, "price"]))
    ws.cell(row, head.index("QTY_FILLED") + 1, 2)
    ws.cell(row, head.index("THESIS_NOTE") + 1, "test thesis")
    if not isinstance(ws.cell(row, head.index("catastrophe_stop") + 1).value, (int, float)):
        ws.cell(row, head.index("catastrophe_stop") + 1, float(dec.loc[0, "price"]) * 0.95)
    if ws.cell(row, head.index("signal") + 1).value not in ("BUY", "SELL"):
        ws.cell(row, head.index("signal") + 1, "BUY")
    ws.cell(row, head.index("contract") + 1, "MZC")
    wb.save(fiche)

    res2 = run_daily(settings, asof="2026-11-23", use_ib=False, update=False)
    trades = store.read_table("trades")
    assert len(trades) == 1 and trades.iloc[0]["status"] == "OPEN"
    assert not res2["positions"].empty and res2["positions"].iloc[0]["qty_open"] == 2

    # running again must not duplicate the trade (ingested once)
    run_daily(settings, asof="2026-11-23", use_ib=False, update=False)
    assert len(store.read_table("trades")) == 1

    # --- the user exits in the Positions sheet of the latest fiche
    wb = load_workbook(res2["fiche"])
    ws = wb["Positions"]
    head = [c.value for c in ws[1]]
    ws.cell(2, head.index("EXIT_QTY") + 1, 2)
    ws.cell(2, head.index("EXIT_PRICE") + 1, float(dec.loc[0, "price"]) + 10)
    wb.save(res2["fiche"])
    run_daily(settings, asof="2026-11-24", use_ib=False, update=False)
    t = store.read_table("trades").iloc[0]
    assert t["status"] == "CLOSED"
    assert float(t["realized_pnl_usd"]) == 10 * 5.0 * 2  # 10 cents x 5 USD x 2 micro

    hist = pd.ExcelFile(settings.path("history_workbook")).sheet_names
    for s in ("Signals_history", "Trades", "Fills", "Equity", "Performance", "Betas_history"):
        assert s in hist
    sig = pd.read_excel(settings.path("history_workbook"), sheet_name="Signals_history")
    assert set(sig["run_date"].astype(str)) >= {"2026-11-20", "2026-11-23", "2026-11-24"}


def test_backtest_runs(project):
    settings, store = project
    mem = MemoryStore(store)
    r = run_backtest(mem, settings, "corn", "2025-06-01", "2025-09-30", BTConfig(z_entry=1.0), every=5, progress=False)
    assert len(r["equity"]) > 50
    assert r["equity"].notna().all()
