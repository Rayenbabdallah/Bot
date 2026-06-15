"""Economic calendar news filter.

Flags bars that fall inside a blackout window around high-impact USD
releases (NFP, CPI, FOMC, PPI, PCE, etc.). The calendar is a simple CSV
that the user maintains/updates - see data_files/news_calendar_sample.csv
for the expected format. No live calendar API is wired up (most require
paid keys); plug one in here if you have access to one.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_calendar(path: str | Path) -> pd.DataFrame:
    """Load a news calendar CSV with columns: datetime_utc, currency, impact, event."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["datetime_utc", "currency", "impact", "event"])

    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def news_blackout_mask(index: pd.DatetimeIndex, calendar: pd.DataFrame,
                        minutes_before: int, minutes_after: int,
                        currencies: tuple[str, ...] = ("USD",),
                        impacts: tuple[str, ...] = ("high",)) -> pd.Series:
    """Return a boolean Series aligned to `index`, True where trading should be blocked."""
    mask = pd.Series(False, index=index)
    if calendar.empty:
        return mask

    events = calendar[
        calendar["currency"].isin(currencies)
        & calendar["impact"].str.lower().isin([i.lower() for i in impacts])
    ]

    before = pd.Timedelta(minutes=minutes_before)
    after = pd.Timedelta(minutes=minutes_after)
    for event_time in events["datetime_utc"]:
        window = (index >= event_time - before) & (index <= event_time + after)
        mask |= window
    return mask
