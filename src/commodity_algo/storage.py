"""Bitemporal data store (SQLite).

Strategy section 1.2: "Le stockage horodaté est la brique la plus importante."
Every value is stored with TWO dates:
  * period       - the date the value describes (e.g. the 2026-07 July temperature)
  * published_at - when the value became known (e.g. the WASDE release date)

A revision never overwrites history: it is a new row with a later published_at.
`get_series(..., as_of=d)` returns only what was known on date d, so backtests
can never "know the future".

The same database also accumulates every daily output (signals, positions,
journal, betas). The Excel files are views of this store, and the store can be
rebuilt from the Excel archive (see excel_io.ingest_archive).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    series_id    TEXT NOT NULL,
    period       TEXT NOT NULL,
    value        REAL,
    published_at TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    source       TEXT,
    PRIMARY KEY (series_id, period, published_at)
);
CREATE INDEX IF NOT EXISTS ix_obs_series ON observations(series_id, period);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _iso(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Bitemporal observations
    # ------------------------------------------------------------------
    def upsert_observations(
        self,
        series_id: str,
        values: pd.Series,
        published_at: pd.Series | str | None = None,
        source: str = "",
    ) -> int:
        """Store a series. Only NEW information creates rows:
        - a period never seen before, or
        - a period whose latest known value differs (a revision).
        Returns the number of rows inserted.
        """
        values = values.dropna()
        if values.empty:
            return 0
        fetched = _utcnow()
        if published_at is None:
            pub = pd.Series(fetched, index=values.index)
        elif isinstance(published_at, str):
            pub = pd.Series(published_at, index=values.index)
        else:
            pub = published_at.reindex(values.index).fillna(fetched)

        latest = self._latest_values(series_id)
        rows = []
        for period, val in values.items():
            p = pd.Timestamp(period).strftime("%Y-%m-%d")
            prev = latest.get(p)
            if prev is not None and abs(prev - float(val)) <= 1e-12 * max(1.0, abs(prev)):
                continue
            rows.append((series_id, p, float(val), _iso(pub.loc[period]), fetched, source))
        if not rows:
            return 0
        with self.connect() as con:
            con.executemany(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?)", rows
            )
        return len(rows)

    def _latest_values(self, series_id: str) -> dict[str, float]:
        q = """
        SELECT o.period, o.value FROM observations o
        JOIN (SELECT period, MAX(published_at) AS mp FROM observations
              WHERE series_id=? GROUP BY period) m
          ON o.period=m.period AND o.published_at=m.mp
        WHERE o.series_id=?
        """
        with self.connect() as con:
            return {p: v for p, v in con.execute(q, (series_id, series_id)).fetchall()}

    def get_series(self, series_id: str, as_of=None) -> pd.Series:
        """Latest known value per period, using only data published <= as_of."""
        params: list = [series_id]
        cond = ""
        if as_of is not None:
            cond = " AND published_at <= ?"
            params.append(pd.Timestamp(as_of).strftime("%Y-%m-%dT23:59:59"))
        q = f"""
        SELECT period, value, published_at FROM observations
        WHERE series_id=?{cond}
        """
        with self.connect() as con:
            df = pd.read_sql_query(q, con, params=params)
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        df = df.sort_values(["period", "published_at"]).groupby("period").tail(1)
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["period"]), name=series_id)
        return s.sort_index()

    def series_ids(self) -> list[str]:
        with self.connect() as con:
            return [r[0] for r in con.execute("SELECT DISTINCT series_id FROM observations")]

    def data_freshness(self) -> pd.DataFrame:
        q = """SELECT series_id, MAX(period) AS last_period, MAX(published_at) AS last_published,
                      MAX(fetched_at) AS last_fetched, COUNT(*) AS n_rows, MAX(source) AS source
               FROM observations GROUP BY series_id ORDER BY series_id"""
        with self.connect() as con:
            return pd.read_sql_query(q, con)

    # ------------------------------------------------------------------
    # Generic accumulating tables (signals, journal, positions, betas...)
    # ------------------------------------------------------------------
    def read_table(self, table: str) -> pd.DataFrame:
        with self.connect() as con:
            exists = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table}"', con)

    def upsert_table(self, table: str, df: pd.DataFrame, keys: Iterable[str]) -> None:
        """Insert or replace rows identified by `keys`. Tolerates new columns
        (schema evolution) by rewriting the table, which is small."""
        if df is None or df.empty:
            return
        keys = list(keys)
        old = self.read_table(table)
        new = df.copy()
        for c in new.columns:
            if pd.api.types.is_datetime64_any_dtype(new[c]):
                new[c] = new[c].dt.strftime("%Y-%m-%d")
        if not old.empty:
            for k in keys:
                old[k] = old[k].astype(str)
            new_keys = new[keys].astype(str)
            merged_old = old.merge(new_keys.drop_duplicates(), on=keys, how="left", indicator=True)
            old = old[(merged_old["_merge"] == "left_only").values]
            out = pd.concat([old, new.astype({k: str for k in keys})], ignore_index=True)
        else:
            out = new.astype({k: str for k in keys})
        with self.connect() as con:
            out.to_sql(table, con, if_exists="replace", index=False)

    def delete_rows(self, table: str, where: str, params: tuple = ()) -> None:
        with self.connect() as con:
            con.execute(f'DELETE FROM "{table}" WHERE {where}', params)
