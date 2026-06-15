"""Shared test fixtures: synthetic OHLCV data for offline testing without network access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate a synthetic XAU/USD-like 15-minute OHLCV series with a trend
    plus noise, spanning several weeks, so strategy/backtest tests don't
    depend on network access.
    """
    rng = np.random.default_rng(42)
    n_bars = 4000
    index = pd.date_range("2024-01-02", periods=n_bars, freq="15min", tz="UTC")

    # Drift + noise random walk around a gold-like price level.
    drift = 0.0005
    noise = rng.normal(0, 1.0, n_bars)
    close = 2000 + np.cumsum(drift + noise)
    close = np.clip(close, 1500, 2500)

    high = close + np.abs(rng.normal(0, 0.8, n_bars))
    low = close - np.abs(rng.normal(0, 0.8, n_bars))
    open_ = close + rng.normal(0, 0.5, n_bars)
    open_ = np.clip(open_, low, high)
    volume = rng.integers(100, 1000, n_bars)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    return df
