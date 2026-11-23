"""Interactive Brokers connection (TWS or IB Gateway, paper or live).

Read-only by default: prices, account value, positions, executions.
`stage_bracket()` can create orders with transmit=False: they appear in TWS
and YOU click "Transmit" - the strategy keeps the last step manual (section 1.2).

Setup in TWS: File > Global Configuration > API > Settings
  [x] Enable ActiveX and Socket Clients
  Socket port 7497 (paper) / 7496 (live)
  [ ] Read-Only API  -> leave ticked unless you want staged orders
  Trusted IPs: 127.0.0.1
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ContractInfo:
    symbol: str
    local_symbol: str
    expiry: pd.Timestamp
    con_id: int
    multiplier: float
    exchange: str


class IBClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
                 market_data_type: int = 3, readonly: bool = True):
        from ib_async import IB  # imported lazily: optional dependency
        self.ib = IB()
        self.host, self.port, self.client_id = host, port, client_id
        self.market_data_type = market_data_type
        self.readonly = readonly

    # ------------------------------------------------------------------
    def connect(self, timeout: float = 10) -> "IBClient":
        self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout,
                        readonly=self.readonly)
        self.ib.reqMarketDataType(self.market_data_type)
        return self

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    # ------------------------------------------------------------------
    def account(self) -> dict:
        accts = self.ib.managedAccounts()
        acct = accts[0] if accts else ""
        vals = {v.tag: v for v in self.ib.accountSummary(acct)}
        def num(tag):
            v = vals.get(tag)
            return float(v.value) if v is not None and v.value not in ("", None) else np.nan
        return {
            "account": acct,
            "is_paper": acct.startswith("D"),  # paper accounts are "DU..."
            "net_liquidation": num("NetLiquidation"),
            "available_funds": num("AvailableFunds"),
            "init_margin": num("InitMarginReq"),
            "maint_margin": num("MaintMarginReq"),
            "cash": num("TotalCashValue"),
        }

    # ------------------------------------------------------------------
    def futures_chain(self, symbol: str, exchange: str) -> list[ContractInfo]:
        from ib_async import Future
        details = self.ib.reqContractDetails(Future(symbol=symbol, exchange=exchange))
        out = []
        for d in details:
            c = d.contract
            exp = pd.Timestamp(c.lastTradeDateOrContractMonth[:8])
            out.append(ContractInfo(c.symbol, c.localSymbol, exp, c.conId,
                                    float(c.multiplier or 1), c.exchange))
        return sorted(out, key=lambda x: x.expiry)

    def active_contracts(self, symbol: str, exchange: str, roll_bdays: int, asof=None):
        """Front contract not within `roll_bdays` business days of expiry, plus the next one."""
        asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
        chain = [c for c in self.futures_chain(symbol, exchange)
                 if np.busday_count(asof.date(), c.expiry.date()) > roll_bdays]
        if not chain:
            raise RuntimeError(f"No tradable contract found for {symbol}@{exchange}")
        return chain[0], (chain[1] if len(chain) > 1 else None)

    def _contract(self, info: ContractInfo):
        from ib_async import Contract
        c = Contract(conId=info.con_id, exchange=info.exchange)
        self.ib.qualifyContracts(c)
        return c

    def daily_bars(self, info: ContractInfo, duration: str = "2 Y") -> pd.DataFrame:
        from ib_async import util
        bars = self.ib.reqHistoricalData(self._contract(info), endDateTime="", durationStr=duration,
                                         barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
                                         formatDate=1)
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["open", "high", "low", "close", "volume"]]

    def continuous_bars(self, symbol: str, exchange: str, duration: str = "10 Y") -> pd.DataFrame:
        from ib_async import ContFuture, util
        c = ContFuture(symbol=symbol, exchange=exchange)
        self.ib.qualifyContracts(c)
        bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr=duration,
                                         barSizeSetting="1 day", whatToShow="TRADES", useRTH=True)
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")[["open", "high", "low", "close", "volume"]]

    def implied_vol(self, info: ContractInfo, wait: float = 4.0) -> float | None:
        """Implied vol of the options on this future (generic tick 106), or None."""
        try:
            t = self.ib.reqMktData(self._contract(info), genericTickList="106", snapshot=False)
            self.ib.sleep(wait)
            iv = t.impliedVolatility
            self.ib.cancelMktData(t.contract)
            return float(iv) if iv and iv == iv and iv > 0 else None
        except Exception as e:  # pragma: no cover - depends on subscriptions
            log.warning("implied vol unavailable for %s: %s", info.local_symbol, e)
            return None

    # ------------------------------------------------------------------
    def positions(self) -> pd.DataFrame:
        rows = []
        for p in self.ib.positions():
            c = p.contract
            rows.append({"account": p.account, "symbol": c.symbol, "local_symbol": c.localSymbol,
                         "sec_type": c.secType, "expiry": c.lastTradeDateOrContractMonth,
                         "con_id": c.conId, "position": p.position, "avg_cost": p.avgCost,
                         "multiplier": float(c.multiplier or 1)})
        return pd.DataFrame(rows)

    def executions(self) -> pd.DataFrame:
        rows = []
        for f in self.ib.reqExecutions():
            e, c = f.execution, f.contract
            rows.append({"exec_id": e.execId, "time": pd.Timestamp(e.time), "symbol": c.symbol,
                         "local_symbol": c.localSymbol, "side": e.side, "shares": e.shares,
                         "price": e.price, "order_ref": e.orderRef,
                         "commission": f.commissionReport.commission if f.commissionReport else np.nan})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def stage_bracket(self, info: ContractInfo, action: str, qty: int, limit_price: float,
                      stop_price: float, take_profit: float | None, order_ref: str,
                      transmit: bool = False, allow_live: bool = False):
        """Entry limit + protective stop (+ optional take-profit), NOT transmitted by default."""
        if self.readonly:
            raise RuntimeError("Client opened read-only; reconnect with readonly=False to stage orders")
        acct = self.account()
        if not acct["is_paper"] and not allow_live:
            raise RuntimeError("Live account detected: set ib.allow_live_account_orders: true to stage orders")
        contract = self._contract(info)
        tp = take_profit if take_profit is not None else (limit_price * (1.5 if action == "BUY" else 0.5))
        bracket = self.ib.bracketOrder(action, qty, limit_price, tp, stop_price)
        trades = []
        for i, o in enumerate(bracket):
            o.orderRef = order_ref
            o.tif = "GTC"
            o.transmit = transmit and i == len(bracket) - 1
            if take_profit is None and i == 1:
                continue  # skip take-profit leg when the target is dynamic
            trades.append(self.ib.placeOrder(contract, o))
        return trades
