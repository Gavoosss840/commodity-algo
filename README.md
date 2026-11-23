# commodity-algo

Decision-support algorithm for commodity futures, built from the strategy document
*"Algorithme d'aide à la décision sur matières premières"* (31 Oct 2026).

Every day it estimates the **fundamental fair value** of each market from physical data
(stocks/use, yields, weather) and macro factors (dollar, real rate). It measures the gap
with the market price in standard deviations (**z**), applies the strategy's rules
(catalyst, trend and carry filters, sizing at 0.5–1 % risk, three stops, dynamic targets,
option vs future), and writes an **Excel decision fiche**. You place the orders yourself
in Interactive Brokers. Your fills go back into the algorithm through the same Excel
files. The journal, performance and model betas accumulate day after day.

> It is **not** arbitrage. Each position is a statistical bet. Start on the IB **paper**
> account (roadmap stage 0) for at least one full weather season and several WASDE reports.

---

## 1. What is implemented (mapping to the strategy document)

| Strategy section | Implementation |
|---|---|
| 1.2 Time-stamped storage (*la brique la plus importante*) | `storage.py`: bitemporal SQLite store. Each value has a `period` and a `published_at`. `get_series(as_of=…)` never sees the future. Revisions are new rows. |
| 1.2 Fair value, weighted least squares, λ = 0.99/week | `models/wls.py`: WLS with forgetting factor. It shrinks toward the study betas, scaled by the effective sample size. Newey-West standard errors. |
| 1.2 Guardrails (sign flip, beta drift, residuals widening) | `wls.stability_alerts`: alerts in the fiche. Widening residuals suspend the signal. |
| Study §4: corn stage 1, weather → yield | `models/weather_features.py`: July/August extreme heat above the 75th percentile, July Tmax anomaly. Uses observations, then the 16-day forecast, then climatology. |
| Study §4: corn stage 2, stocks/use + dollar + real rate + China imports → real price | `models/grains.py`, with the study betas (−0.40, −1.31, −0.048, +0.025). |
| Study §6: heat-wave example, demand rationing (+15 % is a ceiling) | `stocks_to_use_nowcast`: fixed-point demand rationing. A test reproduces the −550 Mbu example. |
| 1.3 The five rules | `decision.py`: \|z\| ≥ 2, catalyst within 15 business days, trend and carry filters, 2.5 × ATR(20) catastrophe stop, thesis stop, time stop (2 reports), exits at ½ and ⅔ of the gap, 1 ATR trailing stop |
| 1.4 Portfolio limits | 6 % total open risk, 3 % per group, −4 % in the month halves size, −10 % drawdown stops trading, report-day jump allowance, no new position the day before a major report |
| 1.6 Option vs future | Expected move ÷ implied move (IV from IB) ≥ 1.3 gives a call/put spread idea. Livestock uses options only. Never sell naked options. |
| 1.5 Testing without lying | `backtest.py`: walk-forward with as-of data, IB-like costs, 1–2 ticks of slippage, grid with z 1.5/2/2.5 and 0–1 day delay |
| 1.5 Gate-B indicators | `performance.py`: Sharpe, max drawdown, win rate, expectancy in R, correlation with the S&P 500 |
| Part 3: start with micro corn | `config/settings.yaml`: corn **enabled**. Soybeans, wheat and WTI are present but disabled until their roadmap stage. Hogs and cattle are placeholders. |

## 2. Accounts and API keys you need

| What | Needed for | Cost | Where |
|---|---|---|---|
| **Interactive Brokers** paper account + TWS or IB Gateway | prices, account value, positions, implied vol, optional staged orders | free (paper) | IB Client Portal |
| CBOT market data (level 1) | real-time prices | ~1.25 USD/month (non-pro) | IB > Settings > Market Data Subscriptions. *Delayed data (free) is enough for an end-of-day fiche: `market_data_type: 3`.* |
| **USDA FAS API key** | WASDE balance sheets (PSD) | free | https://api.data.gov/signup/ → `USDA_FAS_API_KEY` |
| EIA API key | WTI model only (stage 4) | free | https://www.eia.gov/opendata/register.php → `EIA_API_KEY` |
| FRED | dollar, real rate, CPI, VIX… | free, **no key** | – |
| Open-Meteo | daily weather and 16-day forecasts | free **for non-commercial use**; paid key for commercial use | https://open-meteo.com |
| GitHub | code and Excel history | free | – |

Copy `.env.example` to `.env` and put the keys there. **Never commit `.env`.**

## 3. Installation (Windows, VS Code)

```bash
git clone https://github.com/<you>/commodity-algo.git
```

```bash
cd commodity-algo
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[dev]"
```

```bash
copy .env.example .env
```

(macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.)

**TWS / IB Gateway**: *File › Global Configuration › API › Settings*. Tick *Enable ActiveX and
Socket Clients*. Set the port to 7497 (paper TWS) or 4002 (paper Gateway). Add 127.0.0.1 as a trusted IP.
Then check the connection and the contract symbols:

```bash
python -m commodity_algo test-ib
```

**First-time backfill** (FRED since 1990, USDA since 1992, weather since 1960, 10 years of prices):

```bash
python -m commodity_algo update --full
```

```bash
python -m commodity_algo status
```

## 4. Daily workflow

1. After the CBOT close, with TWS running:

   ```bash
   python -m commodity_algo run
   ```

   You can also use `run_daily.bat` with Windows Task Scheduler. It also commits the new Excel files to Git.
2. Open `output/daily/<year>/fiche_<date>.xlsx`:
   * **Decision**: one row per market. `signal` is BUY / SELL / NONE and `reason` says why.
     The row also gives contract, quantity, limit price, catastrophe stop (to place in IB), targets 1 and 2,
     the thesis-stop fair value, the time stop and the risk in USD and %.
   * **Positions**: open trades with today's action (HOLD, TARGET 1, THESIS STOP, TIME STOP…).
   * **Risk**, **Model** (betas, factor contributions, yield and stocks/use nowcast), **Data** (freshness),
     **Calendar**, **Alerts**.
3. If you trade, fill the **yellow columns** and save:
   * Decision: `EXECUTED (Y/N)` = Y, `FILL_PRICE`, `QTY_FILLED`, `THESIS_NOTE`
   * Positions (exits): `EXIT_QTY`, `EXIT_PRICE`
4. The next run reads those cells (only once) and writes them to the journal (`trades` and `fills`).
   It then computes P&L in R multiples and reconciles with your IB positions.
5. `output/history/master_history.xlsx` is rebuilt every run with all the accumulated history:
   Signals_history, Positions_history, Trades, Fills, Equity, Performance (gate B), Betas_history, Runs.

Optional: `python -m commodity_algo run --stage-orders` creates the entry + stop orders in TWS
**without transmitting them** (you click *Transmit*). It refuses live accounts unless you set
`ib.allow_live_account_orders: true`.

## 5. How the algorithm improves over time ("move forward")

* **Data accumulates** in the bitemporal store. Each USDA report adds a true vintage, and each run adds a day.
* **Betas are re-estimated every run** on all the accumulated daily data, using the forgetting factor
  (recent data weighs more). They are shrunk toward the study betas. Because daily levels are strongly
  autocorrelated, the strength of the shrinkage is scaled by the *effective* sample size. The data takes
  over only when it really disagrees with the study. Every beta is saved in `Betas_history`.
* **Alerts** appear when a beta changes sign or leaves its band, and when residuals widen (the signal is then suspended).
* **Your journal** (R multiples, win rate, expectancy) is in `Performance`. It is what the monthly review
  and the gate-B decision rely on.
* **New factors** (soil moisture, crop condition, ethanol, COT…) are added in `settings.yaml` and the
  data modules. They are kept only if they improve out-of-sample accuracy (study protocol).

## 6. Monthly maintenance (10 minutes)

* After each WASDE, `update` refreshes PSD automatically (with the API key). Without a key, add a row to
  `input/wasde_balance.csv`.
* Check `input/calendar.csv` against the USDA calendar. The dates marked `verified=no` follow the usual pattern.
* Check the CBOT daily limit (`jump_allowance`) and the micro-contract specs on cmegroup.com.
* Run a backtest:

  ```bash
  python -m commodity_algo backtest --market corn --start 2019-01-01 --robust --every 5
  ```

## 7. Known limits (read before trusting a number)

* **Historical USDA balance sheets are final values** (PSD), not the vintages known at the time.
  This look-ahead affects calibration and backtests before your first live run. From now on, true
  vintages accumulate. A clean backtest needs the WASDE archive (Cornell/USDA ESMIS), which is the next data task.
* **Historical report dates** in the backtest are approximated (WASDE around the 10th).
* The price model is the study's annual model mapped to daily data, with front-month continuous prices.
  The study puts its out-of-sample error at about 10 %, so |z| ≥ 2 will be rare. That is by design.
* **Analyst consensus is not available for free.** The "surprise" is our nowcast versus USDA's current number.
* Hogs, cattle and the Brent–WTI spread are not modelled yet (study section 6, next steps).
* Open-Meteo's free tier is for non-commercial use.

## 8. Project layout

```
config/settings.yaml      every parameter (risk, rules, markets, betas) - change here, not in code
input/wasde_balance.csv   optional manual WASDE rows (override PSD)
input/calendar.csv        dated catalysts (weekly reports are generated automatically)
src/commodity_algo/
  storage.py              bitemporal store + accumulating tables
  data/                   fred, psd (USDA), weather (Open-Meteo), eia, ib_client, prices (IB / Yahoo)
  models/                 wls, weather_features, grains (two-stage), oil (weekly event, monitor)
  decision.py             entry rules, filters, sizing, stops, instrument choice
  journal.py              trades / fills, daily position management
  performance.py          equity, drawdown state, gate-B metrics
  excel_io.py             daily fiche, Excel->journal feedback loop, master history
  pipeline.py             the daily run
  backtest.py             walk-forward backtest + robustness grid
  cli.py                  command line
output/daily/             one fiche per day (committed to Git)
output/history/           master_history.xlsx (committed to Git)
data/store.sqlite         the store (not committed; rebuild with `rebuild-from-excel` + `update --full`)
tests/                    offline end-to-end tests on a synthetic dataset
```

## 9. Commands

| Command | What it does |
|---|---|
| `python -m commodity_algo run [--no-ib] [--asof YYYY-MM-DD] [--stage-orders]` | daily run |
| `python -m commodity_algo update [--full] [--no-ib]` | data only |
| `python -m commodity_algo backtest --market corn --start 2019-01-01 [--robust] [--every 5]` | walk-forward backtest |
| `python -m commodity_algo journal list\|open\|close …` | manual journal |
| `python -m commodity_algo status` | data freshness and keys |
| `python -m commodity_algo test-ib` | IB connection and contracts |
| `python -m commodity_algo rebuild-from-excel` | rebuild signals, journal and equity from the Excel archive |
| `pytest -q` | tests (offline) |
