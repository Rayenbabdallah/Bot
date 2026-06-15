import pandas as pd

from gold_bot.backtest.engine import run_backtest
from gold_bot.backtest.metrics import compute_metrics
from gold_bot.config import (AccountConfig, Config, CostsConfig, DataConfig,
                              NewsConfig, RiskConfig, StrategyConfig)
from gold_bot.indicators import add_indicators
from gold_bot.strategy.trend_pullback import generate_signals


def make_config(tmp_path) -> Config:
    return Config(
        data=DataConfig(symbol="GC=F", interval="15m", period="60d", cache_path="unused.parquet"),
        strategy=StrategyConfig(
            fast_ema=20, slow_ema=50, trend_ema=200, atr_period=14,
            atr_stop_mult=2.0, reward_risk=2.0, pullback_lookback=5,
            session_start_utc="00:00", session_end_utc="23:59",
            trade_days=[0, 1, 2, 3, 4, 5, 6],
        ),
        risk=RiskConfig(
            risk_per_trade_pct=0.5, daily_loss_limit_pct=5.0,
            daily_circuit_breaker_pct=2.5, max_overall_loss_pct=10.0,
            max_overall_circuit_breaker_pct=8.0, max_consecutive_losses=2,
        ),
        costs=CostsConfig(spread_usd=0.20, slippage_usd=0.05, news_spread_multiplier=10),
        news=NewsConfig(
            calendar_path=str(tmp_path / "no_calendar.csv"),
            blackout_minutes_before=15, blackout_minutes_after=15,
        ),
        account=AccountConfig(starting_equity=100_000.0, contract_size=100),
    )


def test_backtest_runs_and_produces_metrics(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    df = add_indicators(synthetic_ohlcv, fast_ema=cfg.strategy.fast_ema,
                         slow_ema=cfg.strategy.slow_ema, trend_ema=cfg.strategy.trend_ema,
                         atr_period=cfg.strategy.atr_period,
                         pullback_lookback=cfg.strategy.pullback_lookback)
    df = generate_signals(df, cfg.strategy)

    result = run_backtest(df, cfg)
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) == len(df)
    assert not result.equity_curve.isna().any()

    metrics = compute_metrics(result.trades, result.equity_curve, bars_per_year=4 * 24 * 365)
    assert "sharpe" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["max_drawdown_pct"] <= 0


def test_risk_circuit_breaker_caps_overall_loss(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    # Force a very high risk per trade so the overall circuit breaker is
    # exercised within the synthetic dataset.
    cfg.risk.risk_per_trade_pct = 5.0
    df = add_indicators(synthetic_ohlcv, fast_ema=cfg.strategy.fast_ema,
                         slow_ema=cfg.strategy.slow_ema, trend_ema=cfg.strategy.trend_ema,
                         atr_period=cfg.strategy.atr_period,
                         pullback_lookback=cfg.strategy.pullback_lookback)
    df = generate_signals(df, cfg.strategy)

    result = run_backtest(df, cfg)
    min_equity = result.equity_curve.min()
    overall_loss_pct = (cfg.account.starting_equity - min_equity) / cfg.account.starting_equity * 100
    # The circuit breaker should prevent losses from running away far beyond
    # the configured threshold (allow some slack for the loss that trips it).
    assert overall_loss_pct < cfg.risk.max_overall_circuit_breaker_pct + 10
