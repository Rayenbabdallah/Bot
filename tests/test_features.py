from gold_bot.features import FEATURE_COLUMNS, build_features
from gold_bot.indicators import add_indicators


def test_build_features_adds_expected_columns(synthetic_ohlcv):
    df = add_indicators(synthetic_ohlcv, fast_ema=20, slow_ema=50, trend_ema=200,
                         atr_period=14, pullback_lookback=5)
    out = build_features(df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns

    # After warm-up, feature columns should be finite (no inf/NaN explosions).
    warm = out.iloc[300:]
    assert warm[FEATURE_COLUMNS].notna().all().all()
