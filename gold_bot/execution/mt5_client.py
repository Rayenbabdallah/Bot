"""Thin wrapper around the MetaTrader5 Python package.

The `MetaTrader5` package is Windows-only (it talks to a local MT5
terminal) and is not installed in this environment - the import is lazy so
the rest of the codebase (backtesting, ML, etc.) works without it. This
module defines the client interface that `gold_bot.execution.live_runner`
depends on; `tests/test_execution.py` exercises the runner against a
`FakeMT5Client` implementing the same interface.

Usage on a Windows VPS with MT5 installed and logged in:

    from gold_bot.execution.mt5_client import MT5Client
    client = MT5Client()
    client.connect(login=..., password=..., server=...)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TIMEFRAME_MAP = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}

ORDER_TYPE_BUY = "ORDER_TYPE_BUY"
ORDER_TYPE_SELL = "ORDER_TYPE_SELL"


@dataclass
class OrderResult:
    success: bool
    ticket: int | None
    comment: str
    raw: object = None


@dataclass
class ClosedTrade:
    ticket: int
    profit: float
    close_time: pd.Timestamp


class MT5Client:
    """Live MT5 client. Requires the `MetaTrader5` package and a running,
    logged-in MT5 terminal (Windows only)."""

    def __init__(self) -> None:
        self._mt5 = None

    def _module(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as e:
                raise ImportError(
                    "Live execution requires the 'MetaTrader5' package and a "
                    "running MT5 terminal (Windows only). "
                    "Install with: pip install MetaTrader5"
                ) from e
            self._mt5 = mt5
        return self._mt5

    def connect(self, login: int | None = None, password: str | None = None,
                 server: str | None = None, path: str | None = None) -> None:
        mt5 = self._module()
        ok = mt5.initialize(path=path, login=login, password=password, server=server)
        if not ok:
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    def get_ohlcv(self, symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
        mt5 = self._module()
        timeframe = getattr(mt5, TIMEFRAME_MAP[interval])
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("timestamp")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    def get_account_equity(self) -> float:
        mt5 = self._module()
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {mt5.last_error()}")
        return float(info.equity)

    def get_open_positions(self, symbol: str, magic: int) -> list:
        mt5 = self._module()
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == magic]

    def get_closed_trades_since(self, symbol: str, magic: int,
                                  since: pd.Timestamp) -> list[ClosedTrade]:
        mt5 = self._module()
        deals = mt5.history_deals_get(since.to_pydatetime(), pd.Timestamp.utcnow().to_pydatetime(),
                                        group=symbol)
        if deals is None:
            return []
        return [
            ClosedTrade(ticket=d.ticket, profit=d.profit,
                         close_time=pd.to_datetime(d.time, unit="s", utc=True))
            for d in deals if d.magic == magic and d.entry == 1  # entry==1 -> DEAL_ENTRY_OUT
        ]

    def send_market_order(self, symbol: str, direction: int, volume: float,
                            sl: float, tp: float, magic: int,
                            comment: str = "gold_bot") -> OrderResult:
        mt5 = self._module()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, ticket=None, comment="no tick data")

        price = tick.ask if direction == 1 else tick.bid
        order_type = getattr(mt5, ORDER_TYPE_BUY if direction == 1 else ORDER_TYPE_SELL)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(success=success, ticket=getattr(result, "order", None),
                             comment=result.comment, raw=result)

    def close_position(self, position) -> OrderResult:
        mt5 = self._module()
        direction = -1 if position.type == 0 else 1  # flip: close a buy with a sell
        tick = mt5.symbol_info_tick(position.symbol)
        price = tick.bid if direction == -1 else tick.ask
        order_type = getattr(mt5, ORDER_TYPE_SELL if direction == -1 else ORDER_TYPE_BUY)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": 20,
            "magic": position.magic,
            "comment": "gold_bot close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(success=success, ticket=getattr(result, "order", None),
                             comment=result.comment, raw=result)
