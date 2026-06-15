"""Risk management: position sizing and prop-firm style loss circuit breakers.

Implements the "rule of halves" approach recommended for prop evaluations:
risk a small % per trade, and hard-stop trading for the day (or entirely)
well before the prop firm's actual daily/overall loss limits are hit.
"""
from __future__ import annotations

from dataclasses import dataclass

from gold_bot.config import RiskConfig


def position_size(equity: float, risk_per_trade_pct: float, stop_distance: float,
                   contract_size: float) -> float:
    """Return position size in contracts (lots) for a given stop distance (price units).

    risk_amount = equity * risk_per_trade_pct / 100
    size (in oz) = risk_amount / stop_distance
    size (in contracts) = size_in_oz / contract_size
    """
    if stop_distance <= 0:
        return 0.0
    risk_amount = equity * (risk_per_trade_pct / 100.0)
    size_oz = risk_amount / stop_distance
    return size_oz / contract_size


@dataclass
class RiskState:
    """Mutable per-backtest risk state, reset daily as needed."""

    starting_equity: float
    day_start_equity: float
    peak_equity: float
    consecutive_losses: int = 0
    trading_halted_for_day: bool = False
    trading_halted_overall: bool = False

    @classmethod
    def initial(cls, starting_equity: float) -> "RiskState":
        return cls(
            starting_equity=starting_equity,
            day_start_equity=starting_equity,
            peak_equity=starting_equity,
        )

    def new_day(self, equity: float) -> None:
        self.day_start_equity = equity
        self.consecutive_losses = 0
        self.trading_halted_for_day = False

    def update_on_trade_close(self, pnl: float, equity_after: float, cfg: RiskConfig) -> None:
        self.peak_equity = max(self.peak_equity, equity_after)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= cfg.max_consecutive_losses:
            self.trading_halted_for_day = True

        daily_loss_pct = (self.day_start_equity - equity_after) / self.day_start_equity * 100
        if daily_loss_pct >= cfg.daily_circuit_breaker_pct:
            self.trading_halted_for_day = True

        overall_loss_pct = (self.starting_equity - equity_after) / self.starting_equity * 100
        if overall_loss_pct >= cfg.max_overall_circuit_breaker_pct:
            self.trading_halted_overall = True

    def can_trade(self) -> bool:
        return not (self.trading_halted_for_day or self.trading_halted_overall)
