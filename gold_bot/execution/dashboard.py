"""Status reporting: equity, daily-loss budget usage, open risk, spread.

Per the research guide, ops should track equity, the daily-loss budget,
spread, and open risk on a monitoring dashboard. `status_report` returns a
plain dict suitable for printing, logging, or feeding a real dashboard;
`format_report` renders it as human-readable text.
"""
from __future__ import annotations

from gold_bot.config import Config
from gold_bot.execution.live_runner import LiveConfig
from gold_bot.risk import RiskState


def status_report(client, cfg: Config, live_cfg: LiveConfig, state: RiskState) -> dict:
    equity = client.get_account_equity()
    positions = client.get_open_positions(live_cfg.symbol, live_cfg.magic_number)

    daily_pnl = equity - state.day_start_equity
    daily_pnl_pct = daily_pnl / state.day_start_equity * 100 if state.day_start_equity else 0.0
    daily_budget_pct = cfg.risk.daily_circuit_breaker_pct
    daily_budget_used_pct = max(0.0, -daily_pnl_pct) / daily_budget_pct * 100 if daily_budget_pct else 0.0

    overall_pnl_pct = (equity - state.starting_equity) / state.starting_equity * 100 \
        if state.starting_equity else 0.0

    open_risk = sum(
        abs(getattr(p, "price_open", 0.0) - getattr(p, "sl", 0.0)) * getattr(p, "volume", 0.0)
        * cfg.account.contract_size
        for p in positions
    )

    return {
        "equity": equity,
        "starting_equity": state.starting_equity,
        "day_start_equity": state.day_start_equity,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "daily_circuit_breaker_pct": daily_budget_pct,
        "daily_budget_used_pct": daily_budget_used_pct,
        "overall_pnl_pct": overall_pnl_pct,
        "open_positions": len(positions),
        "open_risk_usd": open_risk,
        "consecutive_losses": state.consecutive_losses,
        "trading_halted_for_day": state.trading_halted_for_day,
        "trading_halted_overall": state.trading_halted_overall,
    }


def format_report(report: dict) -> str:
    lines = [
        "--- Gold Bot Status ---",
        f"Equity:              {report['equity']:,.2f}",
        f"Day start equity:    {report['day_start_equity']:,.2f}",
        f"Daily P&L:           {report['daily_pnl']:,.2f} ({report['daily_pnl_pct']:.2f}%)",
        f"Daily loss budget:   {report['daily_budget_used_pct']:.1f}% used "
        f"(breaker at {report['daily_circuit_breaker_pct']:.1f}% daily loss)",
        f"Overall P&L:         {report['overall_pnl_pct']:.2f}%",
        f"Open positions:      {report['open_positions']}",
        f"Open risk:           {report['open_risk_usd']:,.2f} USD",
        f"Consecutive losses:  {report['consecutive_losses']}",
        f"Halted (day/overall): {report['trading_halted_for_day']} / {report['trading_halted_overall']}",
    ]
    return "\n".join(lines)
