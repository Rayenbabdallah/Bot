"""Stage 3 CLI: live/demo trading loop against MetaTrader5.

Requires the `MetaTrader5` package and a running, logged-in MT5 terminal
(Windows only - see gold_bot/execution/mt5_client.py). Run this against a
DEMO account for at least the 2-month forward test the research guide
recommends before any prop evaluation.

Usage:
    python -m gold_bot.cli_live [--config config/config.yaml] [--once] [--status]

Credentials: set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER environment variables
rather than committing them to config.yaml.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd

from gold_bot.config import load_config
from gold_bot.execution.dashboard import format_report, status_report
from gold_bot.execution.live_runner import LiveConfig, reconcile_closed_trades, run_once
from gold_bot.execution.mt5_client import MT5Client
from gold_bot.execution.state_store import load_state, save_state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live/demo MT5 trading loop")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--status", action="store_true", help="Print a status report and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.execution is None:
        raise SystemExit("config.yaml is missing the 'execution' section - see config/config.yaml")

    live_cfg = LiveConfig(
        symbol=cfg.execution.symbol,
        magic_number=cfg.execution.magic_number,
        n_bars=cfg.execution.n_bars,
    )

    client = MT5Client()
    client.connect(
        login=cfg.execution.mt5_login or _env_int("MT5_LOGIN"),
        password=os.environ.get("MT5_PASSWORD"),
        server=cfg.execution.mt5_server or os.environ.get("MT5_SERVER"),
    )

    state, last_reconcile_time = load_state(cfg.execution.state_path, cfg.account.starting_equity)

    try:
        if args.status:
            print(format_report(status_report(client, cfg, live_cfg, state)))
            return

        while True:
            now = pd.Timestamp.utcnow()
            if now.date() != last_reconcile_time.date():
                equity = client.get_account_equity()
                state.new_day(equity)
                logger.info("New trading day, day_start_equity=%.2f", equity)

            last_reconcile_time = reconcile_closed_trades(
                client, cfg, live_cfg, state, last_reconcile_time
            )

            run_once(client, cfg, live_cfg, state)

            save_state(cfg.execution.state_path, state, last_reconcile_time)

            if args.once:
                break
            time.sleep(cfg.execution.poll_interval_seconds)
    finally:
        client.shutdown()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


if __name__ == "__main__":
    main()
