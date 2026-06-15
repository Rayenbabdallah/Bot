"""Walk-forward evaluation for the XGBoost directional overlay.

Splits the dataset into successive (train, test) windows that roll forward
through time (no shuffling - this is time-series data). For each fold, a
fresh model is trained on the training window and evaluated out-of-sample on
the following test window. Out-of-sample predictions from all folds are
concatenated to form a single OOS prediction series spanning the dataset
(minus the initial training window), which can be fed into the backtest
engine or compared against in-sample performance for walk-forward efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from gold_bot.ml.xgboost_model import XGBDirectionalModel


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_accuracy: float
    test_accuracy: float
    predictions: pd.Series  # predicted direction for the test window


def walk_forward_splits(n_samples: int, train_size: int, test_size: int,
                         step_size: int | None = None):
    """Yield (train_idx, test_idx) integer-position slices for rolling walk-forward folds."""
    step_size = step_size or test_size
    start = 0
    while start + train_size + test_size <= n_samples:
        train_idx = slice(start, start + train_size)
        test_idx = slice(start + train_size, start + train_size + test_size)
        yield train_idx, test_idx
        start += step_size


def run_walk_forward(X: pd.DataFrame, y: pd.Series, train_size: int, test_size: int,
                      step_size: int | None = None, min_confidence: float = 0.0,
                      **xgb_params) -> list[FoldResult]:
    """Run rolling walk-forward training/evaluation. Returns one FoldResult per fold."""
    results: list[FoldResult] = []

    for train_idx, test_idx in walk_forward_splits(len(X), train_size, test_size, step_size):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = XGBDirectionalModel.train(X_train, y_train, **xgb_params)

        train_pred = model.predict_direction(X_train, min_confidence=min_confidence)
        test_pred = model.predict_direction(X_test, min_confidence=min_confidence)

        results.append(FoldResult(
            train_start=X_train.index[0],
            train_end=X_train.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1],
            train_accuracy=accuracy_score(y_train, train_pred),
            test_accuracy=accuracy_score(y_test, test_pred),
            predictions=test_pred,
        ))

    return results


def walk_forward_efficiency(results: list[FoldResult]) -> float:
    """Ratio of mean out-of-sample to in-sample accuracy across folds.

    The research guide flags walk-forward efficiency < 50% (i.e. < 0.5) as a
    sign of overfitting - OOS performance falling off a cliff vs IS.
    """
    if not results:
        return float("nan")
    is_acc = np.mean([r.train_accuracy for r in results])
    oos_acc = np.mean([r.test_accuracy for r in results])
    if is_acc == 0:
        return float("nan")
    return oos_acc / is_acc


def concat_oos_predictions(results: list[FoldResult]) -> pd.Series:
    """Concatenate out-of-sample predictions from all folds into one Series."""
    if not results:
        return pd.Series(dtype=int)
    return pd.concat([r.predictions for r in results]).sort_index()
