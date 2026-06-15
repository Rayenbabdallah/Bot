"""Technical indicators and session/time features used by the strategy."""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rolling_high(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).max()


def rolling_low(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).min()


def add_indicators(df: pd.DataFrame, fast_ema: int, slow_ema: int, trend_ema: int,
                    atr_period: int, pullback_lookback: int) -> pd.DataFrame:
    """Return a copy of df with strategy indicator columns added."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], fast_ema)
    out["ema_slow"] = ema(out["close"], slow_ema)
    out["ema_trend"] = ema(out["close"], trend_ema)
    out["atr"] = atr(out, atr_period)

    # Swing high/low over the lookback window, excluding the current bar,
    # used as the pullback-breakout trigger level.
    out["swing_high"] = rolling_high(out["high"].shift(1), pullback_lookback)
    out["swing_low"] = rolling_low(out["low"].shift(1), pullback_lookback)

    out["hour_utc"] = out.index.hour
    out["minute_utc"] = out.index.minute
    out["weekday"] = out.index.weekday  # Mon=0 ... Sun=6
    return out
