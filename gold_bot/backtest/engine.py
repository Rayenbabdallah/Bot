"""Bar-by-bar backtest engine for the trend-pullback strategy.

Design notes (see research guide for rationale):
  - Signals are generated on bar i using data available through bar i's
    close; trades are executed at the OPEN of bar i+1 to avoid lookahead.
  - Stops/targets are checked against each bar's high/low; if both would
    be hit on the same bar, the stop is assumed to hit first (conservative).
  - Realistic transaction costs (spread + slippage) are applied on both
    entry and exit, and widen automatically during news blackout windows.
  - A RiskState enforces prop-firm style daily/overall circuit breakers and
    a max-consecutive-losses pause, vetoing new entries once tripped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gold_bot.config import Config
from gold_bot.news import load_calendar, news_blackout_mask
from gold_bot.risk import RiskState, position_size


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float  # contracts (lots)
    pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestResult:
    """Run the bar-by-bar backtest. `df` must already contain a 'signal' column
    (see gold_bot.strategy.trend_pullback.generate_signals) and indicator columns.
    """
    risk_cfg = cfg.risk
    costs_cfg = cfg.costs
    account_cfg = cfg.account

    calendar = load_calendar(cfg.news.calendar_path)
    news_mask = news_blackout_mask(
        df.index, calendar,
        minutes_before=cfg.news.blackout_minutes_before,
        minutes_after=cfg.news.blackout_minutes_after,
    )

    equity = account_cfg.starting_equity
    risk_state = RiskState.initial(equity)

    trades: list[Trade] = []
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = equity

    position = None  # dict with direction, entry_price, size, stop, target, entry_atr

    half_spread = costs_cfg.spread_usd / 2.0
    slippage = costs_cfg.slippage_usd

    for i in range(1, len(df)):
        ts = df.index[i]
        prev_ts = df.index[i - 1]
        bar = df.iloc[i]
        prev_bar = df.iloc[i - 1]

        if ts.date() != prev_ts.date():
            risk_state.new_day(equity)

        cost_mult = costs_cfg.news_spread_multiplier if news_mask.iloc[i] else 1.0
        side_cost = half_spread * cost_mult + slippage

        # --- Manage open position: check stop/target against this bar's range ---
        if position is not None:
            exit_price = None
            exit_reason = None

            if position["direction"] == 1:
                if bar["low"] <= position["stop"]:
                    exit_price = position["stop"]
                    exit_reason = "stop"
                elif bar["high"] >= position["target"]:
                    exit_price = position["target"]
                    exit_reason = "target"
            else:
                if bar["high"] >= position["stop"]:
                    exit_price = position["stop"]
                    exit_reason = "stop"
                elif bar["low"] <= position["target"]:
                    exit_price = position["target"]
                    exit_reason = "target"

            if exit_price is not None:
                fill_price = exit_price - position["direction"] * side_cost
                price_diff = (fill_price - position["entry_price"]) * position["direction"]
                pnl = price_diff * position["size"] * account_cfg.contract_size
                equity += pnl

                trades.append(Trade(
                    entry_time=position["entry_time"],
                    exit_time=ts,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=fill_price,
                    size=position["size"],
                    pnl=pnl,
                    exit_reason=exit_reason,
                ))
                risk_state.update_on_trade_close(pnl, equity, risk_cfg)
                position = None

        # --- Consider new entry at this bar's open, based on prior bar's signal ---
        if position is None and risk_state.can_trade() and not news_mask.iloc[i]:
            signal = prev_bar["signal"]
            atr_val = prev_bar["atr"]
            if signal != 0 and np.isfinite(atr_val) and atr_val > 0:
                direction = int(signal)
                raw_entry = bar["open"]
                entry_price = raw_entry + direction * side_cost

                stop_distance_price = cfg.strategy.atr_stop_mult * atr_val
                stop_price = entry_price - direction * stop_distance_price
                target_price = entry_price + direction * stop_distance_price * cfg.strategy.reward_risk

                size = position_size(
                    equity=equity,
                    risk_per_trade_pct=risk_cfg.risk_per_trade_pct,
                    stop_distance=stop_distance_price,
                    contract_size=account_cfg.contract_size,
                )

                if size > 0:
                    position = {
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_time": ts,
                        "size": size,
                        "stop": stop_price,
                        "target": target_price,
                        "entry_atr": atr_val,
                    }

        equity_curve.iloc[i] = equity

    # Close any open position at the final bar's close
    if position is not None:
        last_bar = df.iloc[-1]
        fill_price = last_bar["close"] - position["direction"] * side_cost
        price_diff = (fill_price - position["entry_price"]) * position["direction"]
        pnl = price_diff * position["size"] * account_cfg.contract_size
        equity += pnl
        trades.append(Trade(
            entry_time=position["entry_time"],
            exit_time=df.index[-1],
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=fill_price,
            size=position["size"],
            pnl=pnl,
            exit_reason="end_of_data",
        ))
        risk_state.update_on_trade_close(pnl, equity, risk_cfg)
        equity_curve.iloc[-1] = equity

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return BacktestResult(trades=trades_df, equity_curve=equity_curve.ffill())
