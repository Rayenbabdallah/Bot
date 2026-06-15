"""Performance metrics for a backtest trade log and equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series,
                     bars_per_year: float) -> dict:
    """Compute standard performance metrics.

    trades: DataFrame with at least a 'pnl' column (one row per closed trade).
    equity_curve: Series of equity values indexed by timestamp.
    bars_per_year: used to annualize returns/Sharpe from per-bar returns.
    """
    metrics: dict = {}

    metrics["total_trades"] = len(trades)
    if len(trades) == 0:
        metrics.update({
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_pnl": 0.0,
            "avg_win": np.nan,
            "avg_loss": np.nan,
        })
    else:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        metrics["win_rate"] = len(wins) / len(trades)
        gross_profit = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        metrics["profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        metrics["total_pnl"] = trades["pnl"].sum()
        metrics["avg_win"] = wins["pnl"].mean() if len(wins) else np.nan
        metrics["avg_loss"] = losses["pnl"].mean() if len(losses) else np.nan

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe = np.nan
    metrics["sharpe"] = sharpe

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    metrics["max_drawdown_pct"] = drawdown.min() * 100

    n_years = len(equity_curve) / bars_per_year if bars_per_year else np.nan
    if n_years and n_years > 0 and equity_curve.iloc[0] > 0:
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        metrics["cagr_pct"] = (total_return ** (1 / n_years) - 1) * 100 if total_return > 0 else np.nan
    else:
        metrics["cagr_pct"] = np.nan

    metrics["final_equity"] = equity_curve.iloc[-1] if len(equity_curve) else np.nan
    metrics["total_return_pct"] = (
        (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100 if len(equity_curve) else np.nan
    )

    return metrics
