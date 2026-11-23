"""Command line: `commodity-algo <command>` (or `python -m commodity_algo <command>`)."""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import load_settings


def _print_decisions(res):
    d = res["decisions"]
    if d is None or d.empty:
        print("No market analysed.")
        return
    cols = ["market", "status", "signal", "z", "price", "fair_value", "gap_pct", "contract", "quantity",
            "catastrophe_stop", "next_catalyst", "reason"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70, "display.float_format", "{:,.2f}".format):
        print(d[[c for c in cols if c in d]].to_string(index=False))
    if res["positions"] is not None and not res["positions"].empty:
        print("\nOPEN POSITIONS")
        print(res["positions"][["trade_id", "qty_open", "price", "open_R", "action"]].to_string(index=False))


def cmd_run(a, s):
    from .pipeline import run_daily
    res = run_daily(s, asof=a.asof, use_ib=not a.no_ib, update=not a.no_update,
                    stage_orders=a.stage_orders, full_backfill=False)
    _print_decisions(res)
    if res["alerts"]:
        print("\nALERTS")
        for x in res["alerts"]:
            print(" -", x)
    print(f"\nDaily fiche : {res['fiche']}\nHistory     : {res['history']}")


def cmd_update(a, s):
    from .pipeline import connect_ib, update_data
    from .storage import Store
    ib = None if a.no_ib else connect_ib(s)
    try:
        for m in update_data(Store(s.path("database")), s, ib, full=a.full):
            print(m)
    finally:
        if ib is not None:
            ib.disconnect()


def cmd_backtest(a, s):
    from pathlib import Path

    from .backtest import BTConfig, MemoryStore, robustness, run_backtest
    from .storage import Store
    store = Store(s.path("database"))
    out_dir = s.root / "output" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    mem = MemoryStore(store)
    r = run_backtest(mem, s, a.market, a.start, a.end,
                     BTConfig(z_entry=a.z, delay_days=a.delay, capital=s.account["capital_usd"]), every=a.every)
    path = out_dir / f"backtest_{a.market}_{a.start}_{a.end}.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as w:
        r["metrics"].to_excel(w, sheet_name="Metrics", index=False)
        r["trades"].to_excel(w, sheet_name="Trades", index=False)
        r["equity"].rename("equity").to_frame().to_excel(w, sheet_name="Equity")
        if a.robust:
            robustness(store, s, a.market, a.start, a.end, every=a.every).to_excel(w, sheet_name="Robustness", index=False)
    print(r["metrics"].to_string(index=False))
    print(f"\nBacktest workbook: {path}")


def cmd_journal(a, s):
    from . import journal
    from .storage import Store
    store = Store(s.path("database"))
    if a.action == "list":
        t = store.read_table("trades")
        print(t.to_string(index=False) if not t.empty else "journal empty")
    elif a.action == "open":
        tid = journal.open_trade(store, s, market=a.market, contract=a.contract, direction=a.direction,
                                 entry_date=a.date, entry_price=a.price, qty=a.qty, fair_value=a.fair_value,
                                 z=a.z, catastrophe_stop=a.stop, thesis=a.thesis or "")
        print("opened", tid)
    elif a.action == "close":
        journal.close_part(store, a.trade_id, a.date, a.qty, a.price)
        print("recorded exit", a.trade_id)


def cmd_rebuild(a, s):
    """New machine / lost database: rebuild everything that is not re-downloadable
    (signals, your journal, equity) from the Excel archive kept in Git."""
    from .excel_io import ingest_archive, ingest_user_inputs
    from .storage import Store
    store = Store(s.path("database"))
    n = ingest_archive(store, s.path("daily_output_dir"))
    print(f"re-imported {n} decision rows from the daily fiches")
    for m in ingest_user_inputs(store, s, s.path("daily_output_dir")):
        print(" ", m)
    hist = s.path("history_workbook")
    if hist.exists():
        eq = pd.read_excel(hist, sheet_name="Equity")
        if not eq.empty and "date" in eq:
            store.upsert_table("equity", eq.assign(date=pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")), keys=["date"])
            print(f"re-imported {len(eq)} equity rows")
    print("Now run `commodity-algo update --full` to re-download market data.")


def cmd_status(a, s):
    from .storage import Store
    st = Store(s.path("database"))
    fr = st.data_freshness()
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(fr.to_string(index=False) if not fr.empty else "store is empty: run `commodity-algo update --full`")
    for k in ("USDA_FAS_API_KEY", "EIA_API_KEY", "OPEN_METEO_API_KEY"):
        print(f"{k:20s} {'set' if s.secret(k) else 'missing'}")
    print("enabled markets:", ", ".join(s.enabled_markets()))


def cmd_test_ib(a, s):
    from .pipeline import connect_ib
    ib = connect_ib(s)
    if ib is None:
        print("NOT CONNECTED. Start TWS/IB Gateway, enable the API (port", s.ib["port"], ") and retry.")
        sys.exit(1)
    try:
        print(ib.account())
        for m, mc in s.enabled_markets().items():
            for kind, c in mc["contracts"].items():
                try:
                    f, n = ib.active_contracts(c["symbol"], c["exchange"], mc.get("roll_business_days_before_expiry", 10))
                    print(f"{m:10s} {kind:8s} front={f.local_symbol} exp={f.expiry:%Y-%m-%d}  next={n.local_symbol if n else '-'}")
                except Exception as e:
                    print(f"{m:10s} {kind:8s} ERROR {e}")
    finally:
        ib.disconnect()


def main(argv=None):
    p = argparse.ArgumentParser(prog="commodity-algo", description="Commodity fundamental decision-support algorithm")
    p.add_argument("--config", default=None, help="path to settings.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="daily run: update data, models, decisions, Excel fiche")
    r.add_argument("--asof", default=None, help="YYYY-MM-DD (default today)")
    r.add_argument("--no-ib", action="store_true", help="do not connect to Interactive Brokers")
    r.add_argument("--no-update", action="store_true", help="skip data download")
    r.add_argument("--stage-orders", action="store_true", help="create (NOT transmit) bracket orders in TWS")
    r.set_defaults(fn=cmd_run)

    u = sub.add_parser("update", help="download / refresh data only")
    u.add_argument("--full", action="store_true", help="full history backfill (first time)")
    u.add_argument("--no-ib", action="store_true")
    u.set_defaults(fn=cmd_update)

    b = sub.add_parser("backtest", help="walk-forward backtest with as-of data")
    b.add_argument("--market", default="corn")
    b.add_argument("--start", default="2019-01-01")
    b.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    b.add_argument("--z", type=float, default=2.0)
    b.add_argument("--delay", type=int, default=0, help="extra days after the next close before entering")
    b.add_argument("--every", type=int, default=1, help="re-run the model every N days (speed)")
    b.add_argument("--robust", action="store_true", help="add the robustness grid (z 1.5/2/2.5, delay 0/1)")
    b.set_defaults(fn=cmd_backtest)

    j = sub.add_parser("journal", help="manual journal entries")
    j.add_argument("action", choices=["list", "open", "close"])
    j.add_argument("--trade-id")
    j.add_argument("--market")
    j.add_argument("--contract")
    j.add_argument("--direction", choices=["BUY", "SELL"])
    j.add_argument("--date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    j.add_argument("--price", type=float)
    j.add_argument("--qty", type=int)
    j.add_argument("--fair-value", type=float)
    j.add_argument("--z", type=float, default=float("nan"))
    j.add_argument("--stop", type=float)
    j.add_argument("--thesis")
    j.set_defaults(fn=cmd_journal)

    sub.add_parser("rebuild-from-excel", help="re-import the Excel archive into the store").set_defaults(fn=cmd_rebuild)
    sub.add_parser("status", help="data freshness and keys").set_defaults(fn=cmd_status)
    sub.add_parser("test-ib", help="check the Interactive Brokers connection and contracts").set_defaults(fn=cmd_test_ib)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    a.fn(a, load_settings(a.config))


if __name__ == "__main__":
    main()
