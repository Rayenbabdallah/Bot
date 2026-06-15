import pandas as pd

from gold_bot.data.macro import _parse_fred_csv, load_or_fetch_macro
from gold_bot.features import add_macro_features, build_features, macro_feature_columns
from gold_bot.indicators import add_indicators


def test_parse_fred_csv_drops_missing_and_parses_dates():
    raw = pd.DataFrame({
        "DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "DTWEXBGS": ["120.5", ".", "121.0"],
    })
    series = _parse_fred_csv(raw, "dxy")
    assert series.name == "dxy"
    assert len(series) == 2
    assert series.index.tz is not None
    assert list(series.values) == [120.5, 121.0]


def test_load_or_fetch_macro_uses_cache(tmp_path):
    cache_path = tmp_path / "macro.parquet"
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"dxy": [100, 101, 102, 103, 104],
                        "real_yield_10y": [1.5, 1.6, 1.55, 1.6, 1.65]}, index=index)
    df.to_parquet(cache_path)

    loaded = load_or_fetch_macro(cache_path)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_add_macro_features_aligns_and_ffills(synthetic_ohlcv):
    df = add_indicators(synthetic_ohlcv, fast_ema=20, slow_ema=50, trend_ema=200,
                         atr_period=14, pullback_lookback=5)
    df = build_features(df)

    macro_index = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(),
                                  freq="D", tz="UTC")
    macro = pd.DataFrame({
        "dxy": range(len(macro_index)),
        "real_yield_10y": [1.5] * len(macro_index),
    }, index=macro_index)

    out = add_macro_features(df, macro, columns=("dxy", "real_yield_10y"))
    cols = macro_feature_columns(out)
    assert "macro_dxy" in cols
    assert "macro_real_yield_10y" in cols
    # Forward-filled, so no NaNs after the first day's worth of bars.
    assert out["macro_dxy"].iloc[200:].notna().all()


def test_add_macro_features_noop_on_empty(synthetic_ohlcv):
    df = add_indicators(synthetic_ohlcv, fast_ema=20, slow_ema=50, trend_ema=200,
                         atr_period=14, pullback_lookback=5)
    out = add_macro_features(df, pd.DataFrame())
    assert macro_feature_columns(out) == []
