from gold_bot.config import RiskConfig
from gold_bot.risk import RiskState, position_size


def make_risk_cfg(**overrides) -> RiskConfig:
    defaults = dict(
        risk_per_trade_pct=0.5,
        daily_loss_limit_pct=5.0,
        daily_circuit_breaker_pct=2.5,
        max_overall_loss_pct=10.0,
        max_overall_circuit_breaker_pct=8.0,
        max_consecutive_losses=2,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def test_position_size_scales_with_risk_and_stop_distance():
    size = position_size(equity=100_000, risk_per_trade_pct=0.5, stop_distance=5.0,
                          contract_size=100)
    # risk_amount = 500, size_oz = 100, size_lots = 1.0
    assert size == 1.0


def test_position_size_zero_when_stop_distance_zero():
    assert position_size(equity=100_000, risk_per_trade_pct=0.5, stop_distance=0,
                          contract_size=100) == 0.0


def test_daily_circuit_breaker_halts_trading():
    cfg = make_risk_cfg()
    state = RiskState.initial(starting_equity=100_000)
    # Lose 2.5% of starting equity (2500) in one trade -> hits daily breaker.
    state.update_on_trade_close(pnl=-2500, equity_after=97_500, cfg=cfg)
    assert state.trading_halted_for_day is True
    assert state.can_trade() is False


def test_consecutive_losses_halts_trading():
    cfg = make_risk_cfg(max_consecutive_losses=2, daily_circuit_breaker_pct=50)
    state = RiskState.initial(starting_equity=100_000)
    state.update_on_trade_close(pnl=-100, equity_after=99_900, cfg=cfg)
    assert state.can_trade() is True
    state.update_on_trade_close(pnl=-100, equity_after=99_800, cfg=cfg)
    assert state.trading_halted_for_day is True


def test_overall_circuit_breaker_halts_permanently():
    cfg = make_risk_cfg(max_overall_circuit_breaker_pct=8.0, daily_circuit_breaker_pct=50)
    state = RiskState.initial(starting_equity=100_000)
    state.update_on_trade_close(pnl=-8000, equity_after=92_000, cfg=cfg)
    assert state.trading_halted_overall is True
    state.new_day(equity=92_000)
    assert state.can_trade() is False


def test_new_day_resets_daily_state_but_not_overall():
    cfg = make_risk_cfg()
    state = RiskState.initial(starting_equity=100_000)
    state.update_on_trade_close(pnl=-2500, equity_after=97_500, cfg=cfg)
    assert state.trading_halted_for_day is True
    state.new_day(equity=97_500)
    assert state.trading_halted_for_day is False
    assert state.can_trade() is True
