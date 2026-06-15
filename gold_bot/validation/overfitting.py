"""Overfitting diagnostics: Deflated Sharpe Ratio and Probability of Backtest
Overfitting (PBO) via Combinatorially Symmetric Cross-Validation (CSCV).

These follow Bailey & Lopez de Prado's formulations and are the diagnostics
the research guide calls out as credible defenses against overfitting,
alongside walk-forward efficiency (see walk_forward.py).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def _annualize_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return float("nan")
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def expected_max_sharpe(num_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio across `num_trials` independent trials
    with per-trial Sharpe estimation variance `sharpe_variance` (SR_0 in
    Bailey & Lopez de Prado's Deflated Sharpe Ratio).
    """
    if num_trials <= 1:
        return 0.0
    sigma = np.sqrt(sharpe_variance)
    return sigma * (
        (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / num_trials)
        + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (num_trials * np.e))
    )


def deflated_sharpe_ratio(returns: pd.Series, num_trials: int = 1,
                           periods_per_year: float = 252) -> dict:
    """Compute the Deflated Sharpe Ratio (probability the true Sharpe > 0
    after correcting for selection bias across `num_trials` strategy/
    parameter trials and non-normal return moments).

    Returns a dict with the observed Sharpe, the benchmark SR0 used for
    deflation, and the DSR (a probability in [0, 1] - values well above 0.95
    are typically considered evidence the strategy isn't just the best of
    many random/overfit trials).
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 3 or returns.std() == 0:
        return {"sharpe": float("nan"), "sr0": float("nan"), "dsr": float("nan"), "n": n}

    sr = _annualize_sharpe(returns, periods_per_year)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3  # pandas returns excess kurtosis; formula wants raw kurtosis

    # Per-trial Sharpe estimation variance (non-annualized SR used here).
    sr_non_annual = returns.mean() / returns.std()
    sharpe_variance = (1 - skew * sr_non_annual + (kurt - 1) / 4 * sr_non_annual ** 2) / (n - 1)
    sharpe_variance = max(sharpe_variance, 1e-12)

    sr0 = expected_max_sharpe(num_trials, sharpe_variance) * np.sqrt(periods_per_year)

    denom = np.sqrt(1 - skew * sr_non_annual + (kurt - 1) / 4 * sr_non_annual ** 2)
    z = (sr - sr0) / (denom * np.sqrt(periods_per_year)) * np.sqrt(n - 1) if denom > 0 else np.nan
    dsr = norm.cdf(z)

    return {"sharpe": sr, "sr0": sr0, "dsr": dsr, "n": n}


def probability_of_backtest_overfitting(trial_returns: pd.DataFrame, n_splits: int = 10,
                                          metric=_annualize_sharpe,
                                          periods_per_year: float = 252) -> float:
    """Estimate PBO via Combinatorially Symmetric Cross-Validation (CSCV).

    `trial_returns`: DataFrame of per-period returns, one column per
    candidate strategy/parameter configuration ("trial"), same time index
    for all columns.

    The data is split into `n_splits` contiguous blocks; for every way of
    choosing half the blocks as the "training" set (and the rest as
    "testing"), the trial with the best training-set performance is found,
    and we check whether it is *below median* on the test set. PBO is the
    fraction of combinations where the in-sample winner underperforms
    out-of-sample - values above ~0.5 indicate the strategy-selection
    process is likely overfitting to noise.
    """
    if trial_returns.shape[1] < 2:
        raise ValueError("PBO requires at least 2 trials (columns) to compare.")

    n = len(trial_returns)
    block_size = n // n_splits
    if block_size == 0:
        raise ValueError("Not enough data for the requested number of splits.")

    blocks = [trial_returns.iloc[i * block_size:(i + 1) * block_size] for i in range(n_splits)]
    block_indices = range(n_splits)

    overfit_count = 0
    total_count = 0
    for train_blocks in combinations(block_indices, n_splits // 2):
        test_blocks = [b for b in block_indices if b not in train_blocks]
        if not test_blocks:
            continue

        train_data = pd.concat([blocks[i] for i in train_blocks])
        test_data = pd.concat([blocks[i] for i in test_blocks])

        train_perf = train_data.apply(metric, periods_per_year=periods_per_year)
        test_perf = test_data.apply(metric, periods_per_year=periods_per_year)

        best_trial = train_perf.idxmax()
        test_rank = test_perf.rank(pct=True)[best_trial]

        if test_rank < 0.5:
            overfit_count += 1
        total_count += 1

    return overfit_count / total_count if total_count else float("nan")
