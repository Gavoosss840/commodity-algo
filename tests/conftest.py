"""Synthetic but internally consistent dataset: lets the whole pipeline run
offline (no IB, no internet) so every change can be tested."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from commodity_algo.config import load_settings
from commodity_algo.storage import Store

ROOT = Path(__file__).resolve().parents[1]


def _make_store(store: Store, settings, end="2026-11-20", seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", end)
    n = len(days)
    usd = pd.Series(110 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=days)
    rr = pd.Series(np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, n)), -1.5, 3.0), index=days)
    cpi_idx = pd.date_range("2015-01-01", end, freq="MS")
    cpi = pd.Series(240 * (1.0025 ** np.arange(len(cpi_idx))), index=cpi_idx)
    pub_d = pd.Series([d + pd.offsets.BDay(1) for d in days], index=days)
    store.upsert_observations("DTWEXBGS", usd, pub_d, "test")
    store.upsert_observations("DFII10", rr, pub_d, "test")
    store.upsert_observations("CPIAUCSL", cpi, pd.Series([d + pd.DateOffset(months=1, days=14) for d in cpi_idx], index=cpi_idx), "test")
    store.upsert_observations("SP500", pd.Series(4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=days), pub_d, "test")

    # PSD balance by marketing year
    su, china = {}, {}
    for my in range(1995, 2027):
        p = pd.Timestamp(my, 5, 1)
        s = float(np.clip(0.13 + 0.03 * np.sin(my), 0.07, 0.2))
        su[p], china[p] = s, 5.0 + (my - 1995) * 0.6
        bal = {"area": 82.0 + 0.1 * (my - 2000), "yield": 140 + 1.9 * (my - 2000), "ending_stocks": 2000 * s / 0.13,
               "total_use": 2000 / 0.13, "production": 14000.0, "beginning_stocks": 1500.0, "imports": 25.0,
               "stocks_to_use": s}
        for k, v in bal.items():
            store.upsert_observations(f"corn_psd_{k}", pd.Series({p: v}), f"{my}-05-10", "test")
        store.upsert_observations("corn_china_imports", pd.Series({p: china[p]}), f"{my}-05-10", "test")

    # weather: seasonal cycle + noise, 5 points, from 1990
    wdays = pd.date_range("1990-01-01", pd.Timestamp(end) - pd.Timedelta(days=6))
    doy = wdays.day_of_year.values
    for pt in settings.markets["corn"]["weather_points"]:
        tmax = 60 + 25 * np.sin((doy - 105) / 365 * 2 * np.pi) + rng.normal(0, 6, len(wdays))
        prcp = np.clip(rng.gamma(0.5, 0.25, len(wdays)), 0, None)
        pub = pd.Series([d + pd.Timedelta(days=6) for d in wdays], index=wdays)
        store.upsert_observations(f"wx_{pt['name']}_tmax", pd.Series(tmax, index=wdays), pub, "test")
        store.upsert_observations(f"wx_{pt['name']}_prcp", pd.Series(prcp, index=wdays), pub, "test")

    # corn price consistent with the study betas + mean-reverting noise
    my_of = np.array([d.year if d.month >= 5 else d.year - 1 for d in days])
    su_d = np.array([su[pd.Timestamp(m, 5, 1)] for m in my_of])
    ch_d = np.array([china[pd.Timestamp(m, 5, 1)] for m in my_of])
    cpi_d = cpi.reindex(cpi.index.union(days)).ffill().reindex(days).values
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.985 * noise[i - 1] + rng.normal(0, 0.012)
    lnp = (9.0 - 1.31 * np.log(usd.values) - 0.40 * np.log(su_d) - 0.048 * rr.values
           + 0.025 * np.log(ch_d) + noise)
    close = np.exp(lnp) * cpi_d
    close = close / np.median(close) * 450.0  # cents/bu
    hi = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    pubp = pd.Series(days, index=days)
    store.upsert_observations("px_corn_close", pd.Series(close, index=days), pubp, "test")
    store.upsert_observations("px_corn_high", pd.Series(hi, index=days), pubp, "test")
    store.upsert_observations("px_corn_low", pd.Series(lo, index=days), pubp, "test")
    store.upsert_observations("px_corn_volume", pd.Series(1000.0, index=days), pubp, "test")
    store.upsert_observations("px_corn_next_close", pd.Series(close * 1.01, index=days), pubp, "test")


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    shutil.copytree(ROOT / "config", root / "config")
    shutil.copytree(ROOT / "input", root / "input")
    settings = load_settings(root / "config" / "settings.yaml", root=root)
    store = Store(settings.path("database"))
    _make_store(store, settings)
    return settings, store
