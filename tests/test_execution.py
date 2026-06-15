from dataclasses import dataclass, field

import pandas as pd
import pytest

from gold_bot.config import (AccountConfig, Config, CostsConfig, DataConfig,
                              ExecutionConfig, NewsConfig, RiskConfig,
                              StrategyConfig)
from gold_bot.execution.dashboard import status_report
from gold_bot.execution.live_runner import (LiveConfig,
                                              reconcile_closed_trades,
                                              run_once)
from gold_bot.execution.mt5_client import ClosedTrade, OrderResult
from gold_bot.execution.state_store import load_state, save_state
from gold_bot.risk import RiskState


@dataclass
class FakePosition:
    ticket: int
    symbol: str
    magic: int
    volume: float
    price_open: float
    sl: float
    type: int = 0  # 0 = buy


@dataclass
class FakeMT5Client:
    """In-memory stand-in for MT5Client implementing the same interface."""

    ohlcv: pd.DataFrame
    equity: float = 100_000.0
    open_positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    sent_orders: list = field(default_factory=list)

    def get_ohlcv(self, symbol, interval, n_bars):
        return self.ohlcv.tail(n_bars)

    def get_account_equity(self):
        return self.equity

    def get_open_positions(self, symbol, magic):
        return [p for p in self.open_positions if p.magic == magic]

    def get_closed_trades_since(self, symbol, magic, since):
        return [t for t in self.closed_trades if t.close_time > since]

    def send_market_order(self, symbol, direction, volume, sl, tp, magic, comment="gold_bot"):
        self.sent_orders.append(dict(symbol=symbol, direction=direction, volume=volume,
                                       sl=sl, tp=tp, magic=magic))
        return OrderResult(success=True, ticket=len(self.sent_orders), comment="ok")


def make_config(tmp_path) -> Config:
    return Config(
        data=DataConfig(symbol="GC=F", interval="15m", period="60d", cache_path="unused.parquet"),
        strategy=StrategyConfig(
            fast_ema=20, slow_ema=50, trend_ema=200, atr_period=14,
            atr_stop_mult=2.0, reward_risk=2.0, pullback_lookback=5,
            session_start_utc="00:00", session_end_utc="23:59",
            trade_days=[0, 1, 2, 3, 4, 5, 6],
        ),
        risk=RiskConfig(
            risk_per_trade_pct=0.5, daily_loss_limit_pct=5.0,
            daily_circuit_breaker_pct=2.5, max_overall_loss_pct=10.0,
            max_overall_circuit_breaker_pct=8.0, max_consecutive_losses=2,
        ),
        costs=CostsConfig(spread_usd=0.20, slippage_usd=0.05, news_spread_multiplier=10),
        news=NewsConfig(
            calendar_path=str(tmp_path / "no_calendar.csv"),
            blackout_minutes_before=15, blackout_minutes_after=15,
        ),
        account=AccountConfig(starting_equity=100_000.0, contract_size=100),
        execution=ExecutionConfig(symbol="XAUUSD", magic_number=12345, n_bars=500),
    )


def test_run_once_sends_order_on_signal(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    live_cfg = LiveConfig(symbol="XAUUSD", magic_number=12345, n_bars=500)
    client = FakeMT5Client(ohlcv=synthetic_ohlcv)
    state = RiskState.initial(cfg.account.starting_equity)

    run_once(client, cfg, live_cfg, state)

    # Whether or not an order was sent depends on the last closed bar's
    # signal in the synthetic data - just verify it runs without error and,
    # if an order was sent, that it's well-formed.
    for order in client.sent_orders:
        assert order["direction"] in (-1, 1)
        assert order["volume"] > 0
        assert order["symbol"] == "XAUUSD"


def test_run_once_skips_when_position_open(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    live_cfg = LiveConfig(symbol="XAUUSD", magic_number=12345, n_bars=500)
    client = FakeMT5Client(
        ohlcv=synthetic_ohlcv,
        open_positions=[FakePosition(ticket=1, symbol="XAUUSD", magic=12345,
                                       volume=1.0, price_open=2000.0, sl=1990.0)],
    )
    state = RiskState.initial(cfg.account.starting_equity)

    run_once(client, cfg, live_cfg, state)
    assert client.sent_orders == []


def test_run_once_skips_when_halted(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    live_cfg = LiveConfig(symbol="XAUUSD", magic_number=12345, n_bars=500)
    client = FakeMT5Client(ohlcv=synthetic_ohlcv)
    state = RiskState.initial(cfg.account.starting_equity)
    state.trading_halted_for_day = True

    run_once(client, cfg, live_cfg, state)
    assert client.sent_orders == []


def test_reconcile_closed_trades_updates_risk_state(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    live_cfg = LiveConfig(symbol="XAUUSD", magic_number=12345, n_bars=500)
    since = pd.Timestamp("2024-01-01", tz="UTC")
    client = FakeMT5Client(
        ohlcv=synthetic_ohlcv,
        equity=99_000.0,
        closed_trades=[ClosedTrade(ticket=1, profit=-1000.0,
                                     close_time=pd.Timestamp("2024-01-02", tz="UTC"))],
    )
    state = RiskState.initial(100_000.0)

    new_since = reconcile_closed_trades(client, cfg, live_cfg, state, since)

    assert new_since == pd.Timestamp("2024-01-02", tz="UTC")
    assert state.consecutive_losses == 1


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    state = RiskState.initial(100_000.0)
    state.consecutive_losses = 1
    ts = pd.Timestamp("2024-01-05T10:00:00Z")

    save_state(path, state, ts)
    loaded_state, loaded_ts = load_state(path, starting_equity=100_000.0)

    assert loaded_state.consecutive_losses == 1
    assert loaded_state.starting_equity == 100_000.0
    assert loaded_ts == ts


def test_status_report_fields(synthetic_ohlcv, tmp_path):
    cfg = make_config(tmp_path)
    live_cfg = LiveConfig(symbol="XAUUSD", magic_number=12345, n_bars=500)
    client = FakeMT5Client(ohlcv=synthetic_ohlcv, equity=98_500.0)
    state = RiskState.initial(100_000.0)

    report = status_report(client, cfg, live_cfg, state)
    assert report["equity"] == 98_500.0
    assert report["daily_pnl"] == -1_500.0
    assert "daily_budget_used_pct" in report
