"""Fetch macro context series (DXY, US 10Y real yields) from FRED.

FRED (Federal Reserve Economic Data) publishes a no-API-key CSV endpoint:
https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

Default series:
  - DXY proxy: "DTWEXBGS" (Trade Weighted U.S. Dollar Index: Broad, Goods and
    Services), published daily.
  - US 10Y real yield: "DFII10" (10-Year Treasury Inflation-Indexed Security,
    Constant Maturity), the standard "real yield" series referenced in the
    research guide as gold's opportunity-cost driver.

Both are daily series; `gold_bot.features.add_macro_features` forward-fills
them onto intraday bars.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

DEFAULT_SERIES = {
    "dxy": "DTWEXBGS",
    "real_yield_10y": "DFII10",
}


def _parse_fred_csv(raw: pd.DataFrame, column_name: str) -> pd.Series:
    """Normalize a raw FRED CSV DataFrame (DATE + one value column) into a
    UTC-indexed numeric Series named `column_name`, dropping missing values
    (FRED uses "." for missing observations).
    """
    raw = raw.copy()
    raw.columns = [c.strip() for c in raw.columns]
    date_col, value_col = raw.columns[0], raw.columns[1]

    series = pd.to_numeric(raw[value_col], errors="coerce")
    series.index = pd.to_datetime(raw[date_col], utc=True)
    series.name = column_name
    return series.dropna()


def fetch_fred_series(series_id: str, column_name: str) -> pd.Series:
    """Download a single FRED series as a UTC-indexed numeric Series."""
    url = FRED_CSV_URL.format(series_id=series_id)
    raw = pd.read_csv(url)
    return _parse_fred_csv(raw, column_name)


def fetch_macro_data(series: dict[str, str] | None = None) -> pd.DataFrame:
    """Download and combine macro series into a single DataFrame indexed by
    UTC date, with one column per entry in `series` (default: dxy,
    real_yield_10y).
    """
    series = series or DEFAULT_SERIES
    columns = {name: fetch_fred_series(series_id, name) for name, series_id in series.items()}
    return pd.DataFrame(columns)


def load_or_fetch_macro(cache_path: str | Path, series: dict[str, str] | None = None,
                         refresh: bool = False) -> pd.DataFrame:
    """Load macro data from a local parquet cache, fetching it first if absent."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = fetch_macro_data(series)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
