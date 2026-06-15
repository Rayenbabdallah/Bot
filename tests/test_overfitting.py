import numpy as np
import pandas as pd

from gold_bot.validation.overfitting import (deflated_sharpe_ratio,
                                               probability_of_backtest_overfitting)


def test_deflated_sharpe_ratio_basic():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))

    result = deflated_sharpe_ratio(returns, num_trials=1, periods_per_year=252)
    assert np.isfinite(result["sharpe"])
    assert 0.0 <= result["dsr"] <= 1.0

    # More trials -> higher SR0 -> lower (or equal) DSR for the same returns.
    result_many_trials = deflated_sharpe_ratio(returns, num_trials=100, periods_per_year=252)
    assert result_many_trials["sr0"] >= result["sr0"]
    assert result_many_trials["dsr"] <= result["dsr"]


def test_deflated_sharpe_ratio_handles_zero_variance():
    returns = pd.Series([0.0] * 10)
    result = deflated_sharpe_ratio(returns)
    assert np.isnan(result["sharpe"])


def test_pbo_with_noise_trials():
    rng = np.random.default_rng(1)
    n = 1000
    # Several "trials" that are pure noise - PBO should be roughly around 0.5
    # (no trial is consistently best in-sample AND out-of-sample).
    trials = pd.DataFrame({
        f"trial_{i}": rng.normal(0, 0.01, n) for i in range(6)
    })
    pbo = probability_of_backtest_overfitting(trials, n_splits=10)
    assert 0.0 <= pbo <= 1.0


def test_pbo_requires_multiple_trials():
    import pytest
    trials = pd.DataFrame({"only_one": np.random.normal(0, 0.01, 100)})
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trials, n_splits=10)
