"""Live/demo trading loop.

Each call to `run_once`:
  1. Pulls the latest bars from the broker and rebuilds indicators/signals
     (and the ML filter, if enabled).
  2. Reconciles any trades the broker closed since the last check into the
     persisted RiskState (so circuit breakers reflect real fills).
  3. If flat, risk checks pass, and the most recently *closed* bar produced
     an entry signal, sizes and sends a market order with SL/TP attached.
  4. Persists RiskState to disk.

This intentionally mirrors `gold_bot.backtest.engine`: signals are generated
from the last closed bar and acted on "now" (≈ next bar's open in the
backtest), the same ATR-based stop/target/sizing logic is used, and the same
RiskState circuit breakers apply. Keeping the two in sync is what makes the
backtest a meaningful predictor of live behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from gold_bot.config import Config
from gold_bot.features import build_features
from gold_bot.indicators import add_indicators
from gold_bot.ml.dataset import build_dataset
from gold_bot.ml.xgboost_model import XGBDirectionalModel
from gold_bot.news import load_calendar, news_blackout_mask
from gold_bot.risk import RiskState, position_size
from gold_bot.sentiment.service import get_sentiment_score
from gold_bot.strategy.ml_overlay import apply_ml_filter, apply_sentiment_sizing
from gold_bot.strategy.trend_pullback import generate_signals

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    symbol: str          # broker symbol, e.g. "XAUUSD"
    magic_number: int
    n_bars: int = 500     # bars of history to pull each cycle (must exceed trend_ema warm-up)


def reconcile_closed_trades(client, cfg: Config, live_cfg: LiveConfig,
                              state: RiskState, since: pd.Timestamp) -> pd.Timestamp:
    """Fetch trades closed since `since`, fold their P&L into RiskState, and
    return the new "since" timestamp (the latest close time seen, or `since`
    if nothing closed).
    """
    closed = client.get_closed_trades_since(live_cfg.symbol, live_cfg.magic_number, since)
    if not closed:
        return since

    closed = sorted(closed, key=lambda t: t.close_time)
    equity = client.get_account_equity()
    for trade in closed:
        state.update_on_trade_close(trade.profit, equity, cfg.risk)
        logger.info("Reconciled closed trade ticket=%s profit=%.2f", trade.ticket, trade.profit)

    return closed[-1].close_time


def build_signal_frame(df: pd.DataFrame, cfg: Config,
                         ml_model: XGBDirectionalModel | None = None) -> pd.DataFrame:
    """Run indicators + strategy signal generation (+ optional ML filter)."""
    out = add_indicators(
        df,
        fast_ema=cfg.strategy.fast_ema,
        slow_ema=cfg.strategy.slow_ema,
        trend_ema=cfg.strategy.trend_ema,
        atr_period=cfg.strategy.atr_period,
        pullback_lookback=cfg.strategy.pullback_lookback,
    )
    out = generate_signals(out, cfg.strategy)

    if cfg.ml is not None and cfg.ml.enabled and ml_model is not None:
        out = build_features(out)
        X, _ = build_dataset(out, horizon=cfg.ml.horizon,
                               deadband_atr_mult=cfg.ml.deadband_atr_mult)
        if len(X):
            direction = ml_model.predict_direction(X, min_confidence=cfg.ml.min_confidence)
            out = apply_ml_filter(out, direction)

    if cfg.sentiment is not None and cfg.sentiment.enabled:
        score = get_sentiment_score([], method=cfg.sentiment.method)
        out = apply_sentiment_sizing(out, score, threshold=cfg.sentiment.agreement_threshold,
                                        size_boost=cfg.sentiment.size_boost)

    return out


def run_once(client, cfg: Config, live_cfg: LiveConfig, state: RiskState,
              ml_model: XGBDirectionalModel | None = None) -> None:
    """Run one cycle of the live trading loop. Mutates `state` in place.

    Daily rollover (calling `state.new_day(equity)` at the start of a new
    trading day) is the caller's responsibility - see `cli_live.py`'s main
    loop, which tracks the last-seen date and calls `new_day` on change.
    """
    equity = client.get_account_equity()

    df = client.get_ohlcv(live_cfg.symbol, cfg.data.interval, live_cfg.n_bars)
    df = build_signal_frame(df, cfg, ml_model=ml_model)

    last_closed = df.iloc[-2]  # the most recently *closed* bar
    latest_ts = df.index[-2]

    calendar = load_calendar(cfg.news.calendar_path)
    news_mask = news_blackout_mask(
        df.index[-2:], calendar,
        minutes_before=cfg.news.blackout_minutes_before,
        minutes_after=cfg.news.blackout_minutes_after,
    )
    in_news_blackout = bool(news_mask.iloc[-1])

    open_positions = client.get_open_positions(live_cfg.symbol, live_cfg.magic_number)

    if open_positions:
        logger.info("Position already open (%d), skipping entry check", len(open_positions))
        return

    if not state.can_trade():
        logger.info("Risk circuit breaker active, skipping entry check")
        return

    if in_news_blackout:
        logger.info("News blackout active at %s, skipping entry check", latest_ts)
        return

    signal = int(last_closed["signal"])
    atr_val = float(last_closed["atr"])
    if signal == 0 or not (atr_val > 0):
        return

    stop_distance = cfg.strategy.atr_stop_mult * atr_val
    size = position_size(
        equity=equity,
        risk_per_trade_pct=cfg.risk.risk_per_trade_pct,
        stop_distance=stop_distance,
        contract_size=cfg.account.contract_size,
    )
    if "size_multiplier" in last_closed.index:
        size *= float(last_closed["size_multiplier"])

    if size <= 0:
        return

    direction = signal
    # Approximate entry price with the last closed bar's close; the broker's
    # order_send call uses the live bid/ask, so SL/TP are relative offsets.
    ref_price = float(last_closed["close"])
    stop_price = ref_price - direction * stop_distance
    target_price = ref_price + direction * stop_distance * cfg.strategy.reward_risk

    result = client.send_market_order(
        symbol=live_cfg.symbol,
        direction=direction,
        volume=round(size, 2),
        sl=stop_price,
        tp=target_price,
        magic=live_cfg.magic_number,
    )
    logger.info("Order sent: direction=%d size=%.2f sl=%.2f tp=%.2f -> success=%s ticket=%s comment=%s",
                 direction, size, stop_price, target_price, result.success, result.ticket, result.comment)
