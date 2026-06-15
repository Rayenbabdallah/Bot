"""Combine the Stage 1 technical signal with the ML directional filter and
the sentiment sizing overlay (Stage 2).

Per the research guide's recommended architecture:
  - The XGBoost directional model acts as a *hard filter*: a technical entry
    signal is only taken if the model's predicted direction agrees with it
    (an XGBoost prediction of 0/"no edge" or the opposite direction vetoes
    the trade).
  - News sentiment acts as a *sizing* overlay: if sentiment agrees with the
    trade direction (and is non-neutral), the position is sized up by
    `sentiment_size_boost`; otherwise the base size is used. Sentiment does
    not veto trades on its own.
"""
from __future__ import annotations

import pandas as pd

from gold_bot.sentiment.service import sentiment_agrees


def apply_ml_filter(df: pd.DataFrame, ml_direction: pd.Series,
                     signal_col: str = "signal") -> pd.DataFrame:
    """Return a copy of df where `signal_col` is zeroed out wherever the
    ML-predicted direction (`ml_direction`, values in {-1, 0, 1}) does not
    match the technical signal's direction.
    """
    out = df.copy()
    ml_direction = ml_direction.reindex(out.index).fillna(0)

    agrees = (out[signal_col] != 0) & (out[signal_col] == ml_direction)
    out[signal_col] = out[signal_col].where(agrees, 0)
    return out


def apply_sentiment_sizing(df: pd.DataFrame, sentiment_score: float,
                            signal_col: str = "signal", threshold: float = 0.1,
                            size_boost: float = 1.5) -> pd.DataFrame:
    """Add a 'size_multiplier' column: `size_boost` where sentiment agrees
    with the (post-ML-filter) signal direction, 1.0 otherwise.
    """
    out = df.copy()
    multiplier = pd.Series(1.0, index=out.index)

    nonzero = out[signal_col] != 0
    for direction in (-1, 1):
        mask = nonzero & (out[signal_col] == direction)
        if mask.any() and sentiment_agrees(sentiment_score, direction, threshold=threshold) \
                and abs(sentiment_score) >= threshold:
            multiplier[mask] = size_boost

    out["size_multiplier"] = multiplier
    return out
