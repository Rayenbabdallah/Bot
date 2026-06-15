"""Load and validate the bot configuration from YAML."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    symbol: str
    interval: str
    period: str
    cache_path: str


class StrategyConfig(BaseModel):
    fast_ema: int
    slow_ema: int
    trend_ema: int
    atr_period: int
    atr_stop_mult: float
    reward_risk: float
    pullback_lookback: int
    session_start_utc: str
    session_end_utc: str
    trade_days: list[int]


class RiskConfig(BaseModel):
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    daily_circuit_breaker_pct: float
    max_overall_loss_pct: float
    max_overall_circuit_breaker_pct: float
    max_consecutive_losses: int


class CostsConfig(BaseModel):
    spread_usd: float
    slippage_usd: float
    news_spread_multiplier: float


class NewsConfig(BaseModel):
    calendar_path: str
    blackout_minutes_before: int
    blackout_minutes_after: int


class AccountConfig(BaseModel):
    starting_equity: float
    contract_size: float


class WalkForwardConfig(BaseModel):
    train_size: int
    test_size: int
    step_size: int | None = None
    num_trials: int = 1


class MLConfig(BaseModel):
    enabled: bool = False
    horizon: int = 4
    deadband_atr_mult: float = 0.25
    min_confidence: float = 0.4
    walk_forward: WalkForwardConfig


class SentimentConfig(BaseModel):
    enabled: bool = False
    method: str = "lexicon"
    agreement_threshold: float = 0.1
    size_boost: float = 1.5


class Config(BaseModel):
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig
    costs: CostsConfig
    news: NewsConfig
    account: AccountConfig
    ml: MLConfig | None = None
    sentiment: SentimentConfig | None = None


def load_config(path: str | Path = "config/config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
