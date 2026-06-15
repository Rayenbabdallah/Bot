"""Persist RiskState (+ reconciliation watermark) across live-runner restarts.

A live bot can be restarted (crash, VPS reboot, deploy) at any point during
the trading day; without persisting `day_start_equity`,
`consecutive_losses`, the halt flags, and the last-reconciled trade
timestamp, a restart could silently reset the daily circuit breaker or
double-count a closed trade's P&L.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gold_bot.risk import RiskState


def load_state(path: str | Path, starting_equity: float) -> tuple[RiskState, pd.Timestamp]:
    """Return (RiskState, last_reconcile_time). If no state file exists,
    returns a fresh RiskState and `pd.Timestamp.utcnow()` as the watermark.
    """
    path = Path(path)
    if not path.exists():
        return RiskState.initial(starting_equity), pd.Timestamp.utcnow()

    with open(path) as f:
        data = json.load(f)

    last_reconcile_time = pd.Timestamp(data.pop("last_reconcile_time"))
    return RiskState(**data), last_reconcile_time


def save_state(path: str | Path, state: RiskState, last_reconcile_time: pd.Timestamp) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    data["last_reconcile_time"] = last_reconcile_time.isoformat()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
