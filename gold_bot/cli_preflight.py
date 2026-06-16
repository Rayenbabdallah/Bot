"""Pre-flight checklist: validate everything is ready before starting the demo.

Run this before launching cli_live.py. It checks config, data freshness,
news calendar coverage, risk parameters, and optionally the MT5 connection.

Usage:
    python -m gold_bot.cli_preflight [--config config/config.yaml] [--mt5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from gold_bot.config import load_config
from gold_bot.news import load_calendar

PASS = "  [PASS]"
WARN = "  [WARN]"
FAIL = "  [FAIL]"


def check(label: str, passed: bool, detail: str = "", warn_only: bool = False) -> bool:
    tag = PASS if passed else (WARN if warn_only else FAIL)
    line = f"{tag}  {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return passed or warn_only


def run_preflight(config_path: str, check_mt5: bool) -> bool:
    print("\n=== Gold Bot Pre-Flight Checklist ===\n")
    all_ok = True

    # 1. Config loads cleanly
    try:
        cfg = load_config(config_path)
        all_ok &= check("Config loads", True)
    except Exception as e:
        check("Config loads", False, str(e))
        print("\nFATAL: fix config before continuing.")
        return False

    # 2. Data cache exists and is recent
    cache = Path(cfg.data.cache_path)
    if cache.exists():
        age_hours = (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).total_seconds() / 3600
        fresh = age_hours < 25
        all_ok &= check(
            "Price data cache exists and is recent",
            fresh,
            f"{cache.name} — last updated {age_hours:.1f}h ago "
            f"({'OK' if fresh else 'run --refresh-data to update'})",
            warn_only=not fresh,
        )
    else:
        all_ok &= check("Price data cache exists", False,
                         f"{cache} not found — run: python -m gold_bot.cli --refresh-data")

    # 3. News calendar has future events
    calendar = load_calendar(cfg.news.calendar_path)
    now_utc = pd.Timestamp.utcnow()
    if not calendar.empty:
        future = calendar[calendar["datetime_utc"] > now_utc]
        has_future = len(future) > 0
        next_event = future.iloc[0] if has_future else None
        detail = (
            f"{len(future)} upcoming events; next: {next_event['event']} "
            f"at {next_event['datetime_utc']}" if has_future
            else "No future events — calendar needs updating"
        )
        all_ok &= check("News calendar has future events", has_future, detail)
    else:
        all_ok &= check("News calendar loaded", False,
                         f"Calendar empty or missing: {cfg.news.calendar_path}")

    # 4. Risk parameters are prop-firm safe
    r = cfg.risk
    risk_ok = r.risk_per_trade_pct <= 1.0
    all_ok &= check(
        "Risk per trade ≤ 1.0%",
        risk_ok,
        f"risk_per_trade_pct = {r.risk_per_trade_pct}% "
        f"({'OK' if risk_ok else 'TOO HIGH — reduce to 0.3-0.5% for prop challenge'})",
    )

    breaker_ok = r.daily_circuit_breaker_pct < r.daily_loss_limit_pct
    all_ok &= check(
        "Daily circuit breaker < daily loss limit",
        breaker_ok,
        f"circuit_breaker={r.daily_circuit_breaker_pct}% "
        f"limit={r.daily_loss_limit_pct}%",
    )

    overall_ok = r.max_overall_circuit_breaker_pct < r.max_overall_loss_pct
    all_ok &= check(
        "Overall circuit breaker < overall loss limit",
        overall_ok,
        f"circuit_breaker={r.max_overall_circuit_breaker_pct}% "
        f"limit={r.max_overall_loss_pct}%",
    )

    # 5. Execution config present
    if cfg.execution:
        all_ok &= check("Execution config present", True,
                         f"symbol={cfg.execution.symbol} magic={cfg.execution.magic_number}")
        # Check credentials via env
        import os
        has_creds = bool(os.environ.get("MT5_PASSWORD"))
        all_ok &= check(
            "MT5_PASSWORD env var set",
            has_creds,
            "Set with: $env:MT5_PASSWORD='your_password'  (PowerShell)" if not has_creds else "",
            warn_only=not check_mt5,
        )
    else:
        all_ok &= check("Execution config present", False,
                         "Add 'execution:' section to config.yaml (see Stage 3 in README)",
                         warn_only=True)

    # 6. Optional: live MT5 connection
    if check_mt5 and cfg.execution:
        print()
        print("  Checking MT5 connection...")
        try:
            import os
            from gold_bot.execution.mt5_client import MT5Client
            client = MT5Client()
            client.connect(
                login=cfg.execution.mt5_login or (int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None),
                password=os.environ.get("MT5_PASSWORD"),
                server=cfg.execution.mt5_server or os.environ.get("MT5_SERVER"),
            )
            equity = client.get_account_equity()
            client.shutdown()
            all_ok &= check("MT5 connection", True, f"equity = {equity:,.2f}")
        except ImportError:
            all_ok &= check("MT5 connection", False,
                             "MetaTrader5 package not installed — run: pip install MetaTrader5",
                             warn_only=True)
        except Exception as e:
            all_ok &= check("MT5 connection", False, str(e))

    # 7. Demo vs live warning
    print()
    print(f"  {'[INFO]'}  TARGET: run demo forward test for 60-90 days BEFORE prop evaluation.")
    print(f"  {'[INFO]'}  PASS threshold: Sharpe > 0.8, profit factor > 1.3, no circuit breaker triggers.")
    print()

    status = "ALL CHECKS PASSED — ready to start demo" if all_ok else "ISSUES FOUND — fix before going live"
    print(f"=== {status} ===\n")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight checklist before starting the live bot")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--mt5", action="store_true", help="Also test live MT5 connection")
    args = parser.parse_args()

    ok = run_preflight(args.config, check_mt5=args.mt5)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
