"""Build a labeled (features, label) dataset for the directional classifier.

Label definition: sign of the forward N-bar return, thresholded by a small
deadband (in units of ATR) so near-flat moves are labeled "no edge" (0)
rather than forced into a direction - this keeps the classifier from
learning noise on dead bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gold_bot.features import FEATURE_COLUMNS


def build_labels(df: pd.DataFrame, horizon: int = 4, deadband_atr_mult: float = 0.25) -> pd.Series:
    """Return a label Series: 1 = up, -1 = down, 0 = flat/no-edge.

    `horizon` is in bars (e.g. 4 bars of M15 = 1 hour ahead).
    """
    close = df["close"]
    forward_return = close.shift(-horizon) - close
    deadband = df["atr"] * deadband_atr_mult

    label = pd.Series(0, index=df.index)
    label[forward_return > deadband] = 1
    label[forward_return < -deadband] = -1
    return label


def build_dataset(df: pd.DataFrame, horizon: int = 4, deadband_atr_mult: float = 0.25,
                   feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned, with rows containing NaNs (warm-up / horizon tail) dropped."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    y = build_labels(df, horizon=horizon, deadband_atr_mult=deadband_atr_mult)
    X = df[feature_columns]

    valid = X.notna().all(axis=1) & y.notna()
    # Drop the tail where the forward label is undefined (shift(-horizon)).
    valid.iloc[-horizon:] = False

    return X[valid], y[valid]


def class_balance(y: pd.Series) -> dict:
    counts = y.value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in counts.items()}
