import numpy as np

from gold_bot.indicators import add_indicators, atr, ema


def test_ema_matches_pandas_ewm(synthetic_ohlcv):
    series = synthetic_ohlcv["close"]
    result = ema(series, 20)
    expected = series.ewm(span=20, adjust=False).mean()
    assert np.allclose(result, expected)


def test_atr_is_non_negative(synthetic_ohlcv):
    result = atr(synthetic_ohlcv, 14)
    assert (result.dropna() >= 0).all()


def test_add_indicators_columns(synthetic_ohlcv):
    out = add_indicators(synthetic_ohlcv, fast_ema=20, slow_ema=50, trend_ema=200,
                          atr_period=14, pullback_lookback=5)
    for col in ["ema_fast", "ema_slow", "ema_trend", "atr", "swing_high", "swing_low",
                "hour_utc", "minute_utc", "weekday"]:
        assert col in out.columns
