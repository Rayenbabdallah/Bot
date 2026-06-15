"""Stage 2 CLI: train/evaluate the XGBoost directional filter via walk-forward
analysis, optionally apply the sentiment sizing overlay, and backtest the
resulting ML-filtered signal out-of-sample.

Usage:
    python -m gold_bot.cli_ml [--config config/config.yaml] [--refresh-data]
                               [--headlines-file path/to/headlines.txt]

This is a research/evaluation tool, not a production training pipeline: it
prints walk-forward efficiency, per-fold accuracy, the Deflated Sharpe Ratio
of the out-of-sample-filtered backtest, and feature importances from the
final fold's model. Per the research guide: if walk-forward efficiency is
below ~0.5, treat the ML overlay as overfit and don't use it.
"""
from __future__ import annotations

import argparse

from gold_bot.backtest.engine import run_backtest
from gold_bot.backtest.metrics import compute_metrics
from gold_bot.cli import BARS_PER_YEAR
from gold_bot.config import load_config
from gold_bot.data.fetch import load_or_fetch
from gold_bot.data.macro import load_or_fetch_macro
from gold_bot.features import (FEATURE_COLUMNS, add_macro_features,
                                build_features, macro_feature_columns)
from gold_bot.indicators import add_indicators
from gold_bot.ml.dataset import build_dataset
from gold_bot.sentiment.service import get_sentiment_score
from gold_bot.strategy.ml_overlay import apply_ml_filter, apply_sentiment_sizing
from gold_bot.strategy.trend_pullback import generate_signals
from gold_bot.validation.overfitting import deflated_sharpe_ratio
from gold_bot.validation.walk_forward import (concat_oos_predictions,
                                                run_walk_forward,
                                                walk_forward_efficiency)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the ML overlay")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--headlines-file", default=None,
                         help="Optional text file, one news headline per line, "
                              "for the sentiment sizing overlay")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.ml is None:
        raise SystemExit("config.yaml is missing the 'ml' section - see config/config.yaml")

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
    df = build_features(df)

    feature_columns = FEATURE_COLUMNS
    if cfg.macro and cfg.macro.enabled:
        macro = load_or_fetch_macro(cfg.macro.cache_path, series=cfg.macro.series,
                                     refresh=args.refresh_data)
        print(f"Loaded macro data: {list(macro.columns)}, "
              f"{macro.index.min()} to {macro.index.max()}")
        df = add_macro_features(df, macro, columns=tuple(cfg.macro.series.keys()))
        feature_columns = feature_columns + macro_feature_columns(df)

    X, y = build_dataset(df, horizon=cfg.ml.horizon, deadband_atr_mult=cfg.ml.deadband_atr_mult,
                          feature_columns=feature_columns)
    print(f"Dataset: {len(X)} samples, label balance: {y.value_counts(normalize=True).to_dict()}")

    wf_cfg = cfg.ml.walk_forward
    results = run_walk_forward(
        X, y,
        train_size=wf_cfg.train_size,
        test_size=wf_cfg.test_size,
        step_size=wf_cfg.step_size,
        min_confidence=cfg.ml.min_confidence,
    )
    if not results:
        raise SystemExit(
            "Not enough data for the configured walk-forward train/test sizes "
            f"(train_size={wf_cfg.train_size}, test_size={wf_cfg.test_size}, "
            f"available samples={len(X)})."
        )

    print(f"\n--- Walk-Forward Results ({len(results)} folds) ---")
    for i, r in enumerate(results):
        print(f"Fold {i}: train [{r.train_start} -> {r.train_end}] "
              f"test [{r.test_start} -> {r.test_end}] "
              f"train_acc={r.train_accuracy:.3f} test_acc={r.test_accuracy:.3f}")

    wfe = walk_forward_efficiency(results)
    print(f"\nWalk-forward efficiency (OOS/IS accuracy): {wfe:.3f}")
    if wfe < 0.5:
        print("WARNING: efficiency < 0.5 -> per the research guide, treat the "
              "ML overlay as overfit and do not enable it.")

    # --- Apply the ML filter (and optional sentiment sizing) to the OOS region ---
    oos_pred = concat_oos_predictions(results)
    oos_df = df.loc[oos_pred.index].copy()
    oos_df = apply_ml_filter(oos_df, oos_pred)

    if cfg.sentiment and cfg.sentiment.enabled:
        headlines: list[str] = []
        if args.headlines_file:
            with open(args.headlines_file) as f:
                headlines = [line.strip() for line in f if line.strip()]
        score = get_sentiment_score(headlines, method=cfg.sentiment.method)
        print(f"\nSentiment score ({cfg.sentiment.method}, {len(headlines)} headlines): {score:.3f}")
        oos_df = apply_sentiment_sizing(
            oos_df, score,
            threshold=cfg.sentiment.agreement_threshold,
            size_boost=cfg.sentiment.size_boost,
        )

    n_signals_before = (df.loc[oos_pred.index, "signal"] != 0).sum()
    n_signals_after = (oos_df["signal"] != 0).sum()
    print(f"\nOOS entry signals: {n_signals_before} -> {n_signals_after} after ML filter")

    bt_result = run_backtest(oos_df, cfg)
    bars_per_year = BARS_PER_YEAR.get(cfg.data.interval, 252)
    metrics = compute_metrics(bt_result.trades, bt_result.equity_curve, bars_per_year)

    print("\n--- OOS Backtest (ML-filtered) ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:,.4f}")
        else:
            print(f"{key:>20}: {value}")

    returns = bt_result.equity_curve.pct_change().dropna()
    dsr = deflated_sharpe_ratio(returns, num_trials=wf_cfg.num_trials, periods_per_year=bars_per_year)
    print(f"\nDeflated Sharpe Ratio (num_trials={wf_cfg.num_trials}): "
          f"sharpe={dsr['sharpe']:.3f} sr0={dsr['sr0']:.3f} dsr={dsr['dsr']:.3f}")
    if dsr["dsr"] < 0.95:
        print("Note: DSR < 0.95 - the observed Sharpe is not strongly distinguishable "
              "from the best of `num_trials` random/overfit strategies.")

    print("\nFinal-fold feature importances:")
    last_X = X.iloc[-(wf_cfg.train_size):]
    last_y = y.iloc[-(wf_cfg.train_size):]
    from gold_bot.ml.xgboost_model import XGBDirectionalModel
    final_model = XGBDirectionalModel.train(last_X, last_y)
    print(final_model.feature_importances().to_string())


if __name__ == "__main__":
    main()
