# Gold (XAU/USD) Algorithmic Trading Bot — Stage 0/1

This is the foundation stage of a hybrid XAU/USD trading bot, following the
"Building a High-Performing Algorithmic Gold Trading Bot" research guide. It
implements a **baseline, robust strategy + realistic-cost backtester** before
any ML/sentiment overlay is added. The honest goal at this stage is not big
returns — it's a strategy↔risk architecture that survives realistic costs and
respects prop-firm-style risk limits, which the guide identifies as the real
determinant of success.

## What's here

```
gold_bot/
  config.py              # pydantic config loader (config/config.yaml)
  data/fetch.py           # yfinance OHLCV fetch + parquet cache
  indicators.py           # EMA, ATR, swing high/low, session/time features
  strategy/trend_pullback.py  # session-filtered EMA trend-pullback signal
  risk.py                 # position sizing + prop-firm circuit breakers
  news.py                 # economic-calendar news blackout filter
  backtest/engine.py       # bar-by-bar backtest with realistic costs
  backtest/metrics.py      # Sharpe, profit factor, max DD, CAGR, etc.
  cli.py                   # `python -m gold_bot.cli` — runs the whole pipeline
config/config.yaml        # all strategy/risk/cost parameters
data_files/news_calendar_sample.csv  # sample high-impact news calendar
tests/                     # offline tests using synthetic OHLCV data
```

## Strategy (Stage 1 baseline)

Session-filtered EMA trend-pullback/breakout on M15:

- **Trend filter**: only trade when `EMA20 > EMA50 > EMA200` (long) or the
  mirrored downtrend (short).
- **Entry trigger**: price pulls back to touch `EMA20`, then breaks the
  recent swing high/low in the trend direction.
- **Session filter**: London–NY overlap (12:00–16:00 UTC by default),
  Tue–Fri only (configurable).
- **News filter**: no new entries within ±15 min of high-impact USD events
  (NFP/CPI/FOMC — see `data_files/news_calendar_sample.csv`); spread widens
  automatically around those windows for exit-cost realism.
- **Stops/targets**: ATR-based stop (2×ATR default), 2:1 reward:risk target.
- **Risk per trade**: 0.5% of equity, sized from the ATR stop distance.
- **Circuit breakers** (`gold_bot/risk.py`): daily loss circuit breaker at
  2.5% (half of a typical 5% prop daily limit), overall circuit breaker at
  8% (below a typical 10% overall limit), and a pause after 2 consecutive
  losing trades.

All of this is configurable in `config/config.yaml`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the backtest

```bash
python -m gold_bot.cli --refresh-data --plot
```

- `--refresh-data` downloads fresh OHLCV via yfinance (default ticker `GC=F`,
  COMEX gold futures, used as a free XAU/USD proxy — `XAUUSD=X` is also worth
  trying). Without this flag it uses the local parquet cache at
  `data_files/xauusd_15m.parquet`.
- `--plot` saves `equity_curve.png`.

Note: yfinance limits 15-minute intraday history to roughly 60 days, so for
longer backtests use `interval: "1h"` or `"1d"` in `config/config.yaml`
(adjust `period` accordingly), or supply your own OHLCV parquet at
`data.cache_path`.

## Running the tests

```bash
pytest -q
```

Tests run entirely offline against a synthetic OHLCV fixture
(`tests/conftest.py`), so they don't depend on network/data access. They
cover indicator correctness, position sizing, the prop-firm circuit breakers,
and an end-to-end backtest run.

## Interpreting results

`compute_metrics` reports: total trades, win rate, profit factor, Sharpe
(annualized from per-bar returns), max drawdown %, CAGR %, and total
return %. Per the research guide's realism anchors:

- A "good" baseline looks like Sharpe ~0.5–1.0, profit factor 1.5–2.0, max
  drawdown <15%, NOT 100%+ returns at low drawdown (that's an overfitting
  red flag).
- Validate with realistic costs first (`config.costs`: spread ~0.20 USD,
  slippage ~0.05 USD normal hours, news multiplier widens this).
- Before trusting any result, run walk-forward analysis and check for
  parameter sensitivity ("plateau vs cliff") — not yet implemented here.

## Stage 2: ML directional filter + sentiment overlay

```
gold_bot/
  features.py              # feature engineering (returns, vol, EMA distances,
                            #   ATR percentile, Bollinger bandwidth, rolling Hurst,
                            #   cyclical time features); add_macro_features() is an
                            #   extension point for DXY/real-yield series
  ml/dataset.py             # forward-return labeling (+/-/0 with an ATR deadband)
  ml/xgboost_model.py        # XGBDirectionalModel: train/predict_proba/predict_direction
  validation/walk_forward.py # rolling walk-forward train/test folds + efficiency
  validation/overfitting.py  # Deflated Sharpe Ratio + PBO (CSCV)
  sentiment/lexicon.py       # offline keyword-based sentiment scorer (default)
  sentiment/finbert.py       # optional FinBERT scorer (needs transformers+torch)
  sentiment/service.py       # get_sentiment_score() / sentiment_agrees()
  strategy/ml_overlay.py     # apply_ml_filter() + apply_sentiment_sizing()
  cli_ml.py                  # `python -m gold_bot.cli_ml` — walk-forward eval + OOS backtest
```

Architecture, per the guide's recommendation:

- **XGBoost = hard filter.** `apply_ml_filter` zeroes out a technical entry
  signal unless the XGBoost model's predicted direction (trained on
  technical+regime+time features, `gold_bot/features.py`) agrees. The model
  predicts {-1, 0, 1} (down/flat/up) over `ml.horizon` bars with an
  ATR-deadband label (`ml.deadband_atr_mult`) and a confidence threshold
  (`ml.min_confidence`) below which it abstains (0).
- **Sentiment = sizing overlay, not a veto.** `apply_sentiment_sizing` only
  *boosts* position size (`sentiment.size_boost`, default 1.5x) when news
  sentiment agrees with the trade direction beyond
  `sentiment.agreement_threshold`; it never blocks a trade. Defaults to a
  dependency-free lexicon scorer (`gold_bot/sentiment/lexicon.py`); set
  `sentiment.method: "finbert"` and install `transformers`+`torch` for the
  real FinBERT model.
- **Validation is mandatory before trusting any of this.** `cli_ml` runs
  rolling walk-forward folds and reports:
  - **Walk-forward efficiency** (OOS/IS accuracy) — the guide's threshold is
    **< 0.5 means overfit, don't use it**.
  - **Deflated Sharpe Ratio** (`validation/overfitting.py`) — corrects the
    OOS Sharpe for the number of trials/parameter variants tried
    (`ml.walk_forward.num_trials`); DSR should be well above 0.95 before
    trusting the result.
  - **PBO via CSCV** (`probability_of_backtest_overfitting`) — pass it a
    DataFrame of per-period returns for multiple parameter variants (one
    column each) when you sweep parameters; not run by default since it
    needs >1 trial.

### Running it

```bash
python -m gold_bot.cli_ml --refresh-data
# with a sentiment overlay, given a text file of headlines (one per line):
python -m gold_bot.cli_ml --headlines-file headlines.txt
```

Set `ml.enabled` / `sentiment.enabled` to `true` in `config/config.yaml` to
document intent to use the overlay (the Stage 1 `cli.py` backtest does not
yet read these flags - `cli_ml.py` is the dedicated evaluation entry point).

On the bundled synthetic (random-walk) data, walk-forward efficiency comes
out ~0.46 and DSR ~0.23 - correctly flagging "don't use this," which is the
expected/healthy outcome for a strategy with no real edge. Re-run against
real historical XAU/USD data to get a meaningful read.

## Macro features (DXY, US 10Y real yields)

```
gold_bot/data/macro.py   # fetch_macro_data(): FRED's no-key CSV endpoint
                          #   (fredgraph.csv) for DXY ("DTWEXBGS") and the
                          #   10Y real yield ("DFII10"), with a parquet cache
gold_bot/features.py     # add_macro_features(): forward-fills daily macro
                          #   series onto intraday bars as macro_<name> and
                          #   macro_<name>_chg columns
                          # macro_feature_columns(): lists the macro_* columns
                          #   present, for appending to FEATURE_COLUMNS
```

Set `macro.enabled: true` in `config/config.yaml` and `cli_ml.py` will fetch
(or load from cache) DXY and the 10Y real yield from FRED, merge them in as
features (`macro_dxy`, `macro_dxy_chg`, `macro_real_yield_10y`,
`macro_real_yield_10y_chg`), and include them in the XGBoost feature set
automatically. Series IDs are configurable under `macro.series` if you want
different FRED series (e.g. "DTWEXM" for the major-currencies dollar index).

As the guide stresses: **the DXY/real-yield ↔ gold relationship is unstable**
(it decouples during safe-haven and central-bank-buying regimes), so these
are inputs to the model, not hard rules - walk-forward validation should
confirm they actually help before relying on them.

FRED's `fredgraph.csv` endpoint requires no API key, but is blocked by this
sandbox's network egress (only `github.com` is allowlisted here) - fetch it
from an environment with normal internet access.

## Stage 3: MT5 execution + demo forward-test loop

```
gold_bot/execution/
  mt5_client.py    # MT5Client: thin wrapper over the MetaTrader5 package
                    #   (Windows-only, lazy import - everything else works without it)
  live_runner.py    # run_once(): rebuilds signals from latest bars, applies the
                    #   same ATR sizing/risk checks as the backtest, sends orders
  state_store.py    # persists RiskState + reconciliation watermark to JSON
                    #   so a restart can't reset the daily circuit breaker
  dashboard.py       # status_report()/format_report(): equity, daily-loss
                      #   budget used, open risk, halts
gold_bot/cli_live.py # `python -m gold_bot.cli_live` — the live/demo loop
```

**Design**: `run_once` mirrors `backtest/engine.py` as closely as possible -
same indicators/signals/ML filter, same ATR-based stop/target/sizing, same
`RiskState` circuit breakers - so the backtest remains a meaningful predictor
of live behavior. SL/TP are sent with the order so the broker enforces them
even if the bot is offline; `reconcile_closed_trades` folds the resulting
P&L from MT5's trade history back into `RiskState` each cycle.

### Running it

Requires Windows with a logged-in MT5 terminal and `pip install MetaTrader5`
(not installed here - this package can't run on Linux). Set credentials via
env vars, not config.yaml:

```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD=...
export MT5_SERVER=YourBroker-Demo

python -m gold_bot.cli_live --status   # one-off status report
python -m gold_bot.cli_live --once     # single cycle (e.g. for cron)
python -m gold_bot.cli_live            # continuous loop (poll_interval_seconds)
```

Configure `execution.symbol` to match your broker's exact XAU/USD symbol
name (e.g. "XAUUSD", "GOLD", "XAUUSD.s") and `execution.magic_number` to a
value unique to this bot so it doesn't manage other EAs' positions.

`tests/test_execution.py` exercises `run_once`/`reconcile_closed_trades`/
`status_report` against a `FakeMT5Client` implementing the same interface,
so the live-loop logic is tested without MT5 installed.

## Roadmap (per the research guide)

- **Stage 0/1**: data + indicators + baseline strategy + risk-aware
  backtester. ✅
- **Stage 2**: ML directional filter (XGBoost) + sentiment sizing overlay +
  walk-forward/Deflated Sharpe/PBO validation. ✅ (validate against real data
  before enabling)
- **Stage 3**: MT5 execution + demo forward-test loop with persisted risk
  state and a status dashboard. ✅ Run on a DEMO account for ≥2 months to
  validate execution/slippage assumptions against the backtest before risking
  an evaluation fee.
- **Stage 4**: Prop-firm evaluation (smallest account first) using
  0.25–0.5% risk per trade.

## Caveats

See the research guide for full context, but in short: there is no durable
predictive edge here — the value is in disciplined risk management, realistic
cost modeling, and prop-rule compliance. Treat any backtest implying 100%+
annual returns at low drawdown as a red flag. This is educational/technical,
not financial advice.
