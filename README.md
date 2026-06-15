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

## Roadmap (per the research guide)

- **Stage 0/1 (this repo so far)**: data + indicators + baseline strategy +
  risk-aware backtester. ✅
- **Stage 2**: ML/sentiment overlay — XGBoost directional filter on
  technical+macro+regime features, FinBERT sentiment agreement filter,
  walk-forward + Deflated Sharpe + PBO validation. Not yet implemented.
- **Stage 3**: ≥2 months demo MT5 forward test to validate
  execution/slippage assumptions against the backtest.
- **Stage 4**: Prop-firm evaluation (smallest account first) using
  0.25–0.5% risk per trade.

## Live execution (not yet implemented)

The architecture is designed so a `gold_bot/execution/` module using the
`MetaTrader5` Python package can be added later: it would consume the same
`signal` + `RiskState` logic from `gold_bot/strategy` and `gold_bot/risk`,
translating accepted signals into MT5 orders, while the backtest engine
remains the source of truth for strategy logic validation.

## Caveats

See the research guide for full context, but in short: there is no durable
predictive edge here — the value is in disciplined risk management, realistic
cost modeling, and prop-rule compliance. Treat any backtest implying 100%+
annual returns at low drawdown as a red flag. This is educational/technical,
not financial advice.
