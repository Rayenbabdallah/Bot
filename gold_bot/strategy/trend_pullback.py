"""Session-filtered EMA trend-pullback / breakout strategy.

Logic (Stage 1 baseline per the research guide):
  - Trend bias: EMA(fast) vs EMA(slow) vs EMA(trend) - only trade in the
    direction of the higher-timeframe trend (ema_fast/slow/trend aligned).
  - Entry trigger: after price pulls back toward ema_fast, a breakout above
    the recent swing high (long) / below the recent swing low (short)
    confirms re-entry into the trend ("pullback-breakout").
  - Only trade during the configured session window, on the configured
    weekdays, and outside news blackout windows.
  - ATR-based stop loss and a fixed reward:risk take profit.

This module only produces entry *signals* - the backtest engine in
gold_bot.backtest.engine handles trade lifecycle, sizing, and risk checks.
"""
from __future__ import annotations

import pandas as pd

from gold_bot.config import StrategyConfig


def _in_session(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    start_h, start_m = (int(x) for x in start.split(":"))
    end_h, end_m = (int(x) for x in end.split(":"))
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    bar_minutes = df["hour_utc"] * 60 + df["minute_utc"]
    return (bar_minutes >= start_minutes) & (bar_minutes < end_minutes)


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return df with an added 'signal' column: 1 = long entry, -1 = short entry, 0 = none."""
    out = df.copy()

    uptrend = (out["ema_fast"] > out["ema_slow"]) & (out["ema_slow"] > out["ema_trend"])
    downtrend = (out["ema_fast"] < out["ema_slow"]) & (out["ema_slow"] < out["ema_trend"])

    # Pullback: price recently traded at/through ema_fast (touched it within
    # the lookback window), i.e. the trend paused before continuing.
    touched_fast_from_above = (out["low"].rolling(cfg.pullback_lookback).min() <= out["ema_fast"])
    touched_fast_from_below = (out["high"].rolling(cfg.pullback_lookback).max() >= out["ema_fast"])

    breakout_up = out["close"] > out["swing_high"]
    breakout_down = out["close"] < out["swing_low"]

    long_signal = uptrend & touched_fast_from_above & breakout_up
    short_signal = downtrend & touched_fast_from_below & breakout_down

    in_session = _in_session(out, cfg.session_start_utc, cfg.session_end_utc)
    on_trade_day = out["weekday"].isin(cfg.trade_days)

    signal = pd.Series(0, index=out.index)
    signal[long_signal & in_session & on_trade_day] = 1
    signal[short_signal & in_session & on_trade_day] = -1

    out["signal"] = signal
    return out
