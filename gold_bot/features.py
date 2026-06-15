"""Feature engineering for the ML directional-filter overlay (Stage 2).

Builds a feature matrix from OHLCV + the Stage 1 indicator columns:
  - multi-lag returns and realized volatility (momentum/mean-reversion cues)
  - normalized distance-to-EMA features (trend/pullback structure)
  - ATR percentile and Bollinger-bandwidth (volatility regime)
  - a rolling Hurst exponent (trending vs mean-reverting regime)
  - session/time-of-day/day-of-week features

Macro features (DXY, US 10Y real yields) are intentionally left as an
extension point - `add_macro_features` merges in an external series if one
is supplied, but no live macro fetch is wired up here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_hurst(series: pd.Series, window: int) -> pd.Series:
    """Approximate rolling Hurst exponent via rescaled-range (R/S) analysis.

    H ~ 0.5 -> random walk; H > 0.5 -> trending; H < 0.5 -> mean-reverting.
    This is a simplified, vectorization-friendly estimate intended as a
    regime feature, not a precise statistical estimator.
    """
    log_returns = np.log(series / series.shift(1))

    def hurst_of_window(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if len(x) < 8:
            return np.nan
        mean = x.mean()
        deviations = np.cumsum(x - mean)
        r = deviations.max() - deviations.min()
        s = x.std()
        if s == 0 or r == 0:
            return 0.5
        return np.log(r / s) / np.log(len(x))

    return log_returns.rolling(window).apply(hurst_of_window, raw=True)


def build_features(df: pd.DataFrame, return_lags: tuple[int, ...] = (1, 2, 3, 5, 10),
                    vol_window: int = 20, hurst_window: int = 50,
                    atr_pct_window: int = 100) -> pd.DataFrame:
    """Return a copy of df with additional ML feature columns.

    Expects df to already have the Stage 1 indicator columns from
    `gold_bot.indicators.add_indicators` (ema_fast/slow/trend, atr, etc.).
    """
    out = df.copy()
    close = out["close"]

    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    for lag in return_lags:
        out[f"return_lag_{lag}"] = close.pct_change(lag)

    out["volatility"] = log_ret.rolling(vol_window).std()

    # Trend structure: normalized distance of price from each EMA, and
    # the EMA stack ordering (proxy for trend alignment).
    out["dist_ema_fast"] = (close - out["ema_fast"]) / out["atr"]
    out["dist_ema_slow"] = (close - out["ema_slow"]) / out["atr"]
    out["dist_ema_trend"] = (close - out["ema_trend"]) / out["atr"]
    out["ema_stack"] = np.sign(out["ema_fast"] - out["ema_slow"]) + \
        np.sign(out["ema_slow"] - out["ema_trend"])

    # Volatility regime: ATR percentile rank over a long lookback, and
    # Bollinger bandwidth as a squeeze/expansion proxy.
    out["atr_pct_rank"] = out["atr"].rolling(atr_pct_window).rank(pct=True)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_bandwidth"] = (4 * std20) / sma20

    # Regime: rolling Hurst exponent (trending vs mean-reverting).
    out["hurst"] = _rolling_hurst(close, hurst_window)

    # Time features as cyclical encodings (better for tree models than raw ints).
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_utc"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_utc"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)

    return out


def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame,
                        columns: tuple[str, ...] = ("dxy", "real_yield_10y")) -> pd.DataFrame:
    """Merge in macro features (e.g. DXY, US 10Y real yield) aligned by timestamp.

    `macro` should be a DataFrame indexed by UTC timestamp with the given
    columns; values are forward-filled to align with intraday bars. No-op if
    `macro` is empty or missing columns.
    """
    out = df.copy()
    if macro is None or macro.empty:
        return out

    available = [c for c in columns if c in macro.columns]
    if not available:
        return out

    aligned = macro[available].reindex(out.index, method="ffill")
    for col in available:
        out[f"macro_{col}"] = aligned[col]
        out[f"macro_{col}_chg"] = aligned[col].pct_change()
    return out


FEATURE_COLUMNS = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5", "return_lag_10",
    "volatility",
    "dist_ema_fast", "dist_ema_slow", "dist_ema_trend", "ema_stack",
    "atr_pct_rank", "bb_bandwidth", "hurst",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
]


def macro_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the macro_* feature columns present in df (added by
    `add_macro_features`), for appending to FEATURE_COLUMNS when training
    the ML model with macro context.
    """
    return [c for c in df.columns if c.startswith("macro_")]

