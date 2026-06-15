"""CLI entry point: fetch data, generate signals, run the backtest, report results.

Usage:
    python -m gold_bot.cli [--config config/config.yaml] [--refresh-data] [--plot]
"""
from __future__ import annotations

import argparse

from gold_bot.backtest.engine import run_backtest
from gold_bot.backtest.metrics import compute_metrics
from gold_bot.config import load_config
from gold_bot.data.fetch import load_or_fetch
from gold_bot.indicators import add_indicators
from gold_bot.strategy.trend_pullback import generate_signals

BARS_PER_YEAR = {
    "1m": 60 * 24 * 365,
    "5m": 12 * 24 * 365,
    "15m": 4 * 24 * 365,
    "30m": 2 * 24 * 365,
    "1h": 24 * 365,
    "1d": 252,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XAU/USD baseline strategy backtest")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--refresh-data", action="store_true",
                         help="Re-download data instead of using the parquet cache")
    parser.add_argument("--plot", action="store_true", help="Save an equity curve plot")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading data: {cfg.data.symbol} {cfg.data.interval} ({cfg.data.period})")
    df = load_or_fetch(
        symbol=cfg.data.symbol,
        interval=cfg.data.interval,
        period=cfg.data.period,
        cache_path=cfg.data.cache_path,
        refresh=args.refresh_data,
    )
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    df = add_indicators(
        df,
        fast_ema=cfg.strategy.fast_ema,
        slow_ema=cfg.strategy.slow_ema,
        trend_ema=cfg.strategy.trend_ema,
        atr_period=cfg.strategy.atr_period,
        pullback_lookback=cfg.strategy.pullback_lookback,
    )
    df = generate_signals(df, cfg.strategy)
    print(f"Generated {(df['signal'] != 0).sum()} entry signals")

    result = run_backtest(df, cfg)
    bars_per_year = BARS_PER_YEAR.get(cfg.data.interval, 252)
    metrics = compute_metrics(result.trades, result.equity_curve, bars_per_year)

    print("\n--- Backtest Results ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:,.4f}")
        else:
            print(f"{key:>20}: {value}")

    if not result.trades.empty:
        print("\nLast 5 trades:")
        print(result.trades.tail(5).to_string(index=False))

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        result.equity_curve.plot(ax=ax, title="Equity Curve")
        ax.set_ylabel("Equity (USD)")
        fig.tight_layout()
        fig.savefig("equity_curve.png", dpi=120)
        print("\nSaved equity_curve.png")


if __name__ == "__main__":
    main()
