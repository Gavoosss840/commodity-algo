import numpy as np
import pandas as pd
import pytest

from commodity_algo.decision import AccountState, atr, decide
from commodity_algo.models.grains import analyze_grain, stocks_to_use_nowcast
from commodity_algo.models.wls import fit_wls
from commodity_algo.storage import Store


def test_bitemporal_no_lookahead(tmp_path):
    st = Store(tmp_path / "s.sqlite")
    p = pd.Timestamp("2026-09-01")
    st.upsert_observations("x", pd.Series({p: 1.0}), "2026-09-10")
    st.upsert_observations("x", pd.Series({p: 2.0}), "2026-10-09")  # revision
    assert st.get_series("x", as_of="2026-09-05").empty
    assert st.get_series("x", as_of="2026-09-20").iloc[0] == 1.0
    assert st.get_series("x", as_of="2026-10-09").iloc[0] == 2.0
    assert st.get_series("x").iloc[0] == 2.0
    # same value again -> no new row
    assert st.upsert_observations("x", pd.Series({p: 2.0}), "2026-11-01") == 0


def test_wls_recovers_betas():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2000, 2)), columns=["a", "b"])
    y = 1.0 + 2.0 * X["a"] - 0.5 * X["b"] + rng.normal(0, 0.1, 2000)
    r = fit_wls(y, X, lam=1.0)
    assert r.mode == "reestimated"
    assert abs(r.beta["a"] - 2.0) < 0.02 and abs(r.beta["b"] + 0.5) < 0.02
    r2 = fit_wls(y, X, prior=pd.Series({"a": 2.0, "b": -0.5}), fix_betas=True)
    assert abs(r2.intercept - 1.0) < 0.02


def test_stocks_to_use_rationing_is_a_ceiling():
    bal = {"area": 88.0, "yield": 181.0, "ending_stocks": 1850.0, "total_use": 15500.0, "stocks_to_use": 1850 / 15500}
    no_r = stocks_to_use_nowcast(bal, 181.0 * (1 - 0.0345), -0.40, 0.0)
    with_r = stocks_to_use_nowcast(bal, 181.0 * (1 - 0.0345), -0.40, -0.15)
    # study example: -3.45% yield -> about -550 Mbu
    assert -560 < no_r["d_production"] < -540
    assert with_r["su_nowcast"] > no_r["su_nowcast"]  # rationing dampens the drop


def test_corn_analysis_and_decision(project):
    settings, store = project
    an = analyze_grain(store, settings, "corn", "2026-11-20")
    assert an.status in ("OK", "SUSPENDED"), an.alerts
    assert an.model is not None and an.model.n_obs > 100
    assert 200 < an.price < 900 and 200 < an.fair_value < 900
    assert np.isfinite(an.z)
    # the synthetic price was built with the study betas -> re-estimated betas close to them
    if an.model.mode == "reestimated":
        assert abs(an.model.beta["log_usd"] + 1.31) < 0.6
    cal = pd.DataFrame({"date": [pd.Timestamp("2026-12-10")], "event": ["WASDE"], "markets": ["corn"]})
    d = decide(an, settings, AccountState(capital=50000), cal)
    assert d.signal in ("BUY", "SELL", "NONE")
    assert d.catalyst_bdays == 14


def test_forced_signal_sizing(project):
    settings, store = project
    an = analyze_grain(store, settings, "corn", "2026-11-20")
    an.z, an.fair_value = -2.5, an.price * 1.12
    an.gap_pct = np.log(an.fair_value / an.price)
    an.bars = an.bars.copy()
    an.bars["close"] = an.bars["close"].iloc[-1]  # flat: no trend filter
    an.bars = an.bars.drop(columns=["next_close"])
    cal = pd.DataFrame({"date": [pd.Timestamp("2026-12-10")], "event": ["WASDE"], "markets": ["corn"]})
    d = decide(an, settings, AccountState(capital=50000), cal)
    assert d.signal == "BUY", d.reason
    assert d.instrument == "MICRO_FUTURE" and d.quantity >= 1
    assert d.risk_pct <= settings.risk["risk_per_trade_max_pct"] + 1e-9
    assert d.catastrophe_stop < d.entry_limit < d.target_1 < d.target_2
    # drawdown -> stop trading
    d2 = decide(an, settings, AccountState(capital=50000, drawdown_pct=-11), cal)
    assert d2.signal == "NONE" and "STOPPED" in d2.reason


def test_atr():
    df = pd.DataFrame({"high": [11, 12, 13.0], "low": [9, 10, 11.0], "close": [10, 11, 12.0]})
    assert atr(df, 20) == pytest.approx(2.0)
