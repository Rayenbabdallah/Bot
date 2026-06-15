"""Fetch and cache historical XAU/USD OHLCV data.

Uses yfinance as a free data source. "GC=F" (COMEX gold futures) and
"XAUUSD=X" (spot gold FX-style quote) are both reasonable proxies for
XAU/USD price action; futures tend to have longer/cleaner intraday history.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Download OHLCV bars from yfinance.

    Returns a DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume
    """
    import yfinance as yf

    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(
            f"No data returned for symbol={symbol!r} interval={interval!r} "
            f"period={period!r}. yfinance limits intraday history (e.g. 15m "
            f"data to ~60 days) - try a shorter period or a daily interval."
        )

    # yfinance may return a MultiIndex for columns when multiple tickers/fields
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    cols = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    return df


def load_or_fetch(symbol: str, interval: str, period: str, cache_path: str | Path,
                   refresh: bool = False) -> pd.DataFrame:
    """Load OHLCV data from a local parquet cache, fetching it first if absent."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = fetch_ohlcv(symbol, interval, period)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
