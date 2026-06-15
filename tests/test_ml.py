import numpy as np
import pandas as pd

from gold_bot.features import build_features
from gold_bot.indicators import add_indicators
from gold_bot.ml.dataset import build_dataset, class_balance
from gold_bot.ml.xgboost_model import XGBDirectionalModel
from gold_bot.validation.walk_forward import (concat_oos_predictions,
                                                run_walk_forward,
                                                walk_forward_efficiency)


def _make_dataset(synthetic_ohlcv):
    df = add_indicators(synthetic_ohlcv, fast_ema=20, slow_ema=50, trend_ema=200,
                         atr_period=14, pullback_lookback=5)
    df = build_features(df)
    return build_dataset(df, horizon=4, deadband_atr_mult=0.25)


def test_build_dataset_shapes_and_balance(synthetic_ohlcv):
    X, y = _make_dataset(synthetic_ohlcv)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({-1, 0, 1})
    balance = class_balance(y)
    assert abs(sum(balance.values()) - 1.0) < 1e-9


def test_xgb_model_train_predict(synthetic_ohlcv):
    X, y = _make_dataset(synthetic_ohlcv)
    model = XGBDirectionalModel.train(X.iloc[:1000], y.iloc[:1000], n_estimators=10)

    proba = model.predict_proba(X.iloc[1000:1050])
    assert list(proba.columns) == [-1, 0, 1]
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    direction = model.predict_direction(X.iloc[1000:1050], min_confidence=0.0)
    assert set(direction.unique()).issubset({-1, 0, 1})

    importances = model.feature_importances()
    assert len(importances) == len(model.feature_columns)


def test_walk_forward_runs_and_reports_efficiency(synthetic_ohlcv):
    X, y = _make_dataset(synthetic_ohlcv)
    results = run_walk_forward(X, y, train_size=500, test_size=200, step_size=200,
                                n_estimators=10)
    assert len(results) > 0

    wfe = walk_forward_efficiency(results)
    assert np.isfinite(wfe)

    oos = concat_oos_predictions(results)
    assert len(oos) > 0
    assert oos.index.is_monotonic_increasing
