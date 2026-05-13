"""
test_kalman.py — unit tests for the Kalman filter.

Testing:
1. Hedge ratio converges to true β on synthetic data
2. Z-score has mean ≈ 0 and std ≈ 1 after the warmup period
3. Edge cases: short series, no overlapping dates, spread variance reduction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.kalman_filter import fit, zscore


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


def _make_synthetic_pair(
    n: int = 1000,
    beta_true: float = 1.5,
    kappa: float = 0.05,
    sigma_ou: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Y = beta_true·X + OU spread. X is a random walk."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n) * 1.0) + 50.0
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1.0 - kappa) + sigma_ou * rng.standard_normal()
    y = beta_true * x + s
    dates = _make_dates(n)
    return (
        pd.Series(y, index=dates, name="Y"),
        pd.Series(x, index=dates, name="X"),
        s,
    )


class TestKalmanConvergence:
    """After enough observations the hedge ratio should track β_true."""

    def test_converges_to_true_beta(self):
        beta_true = 1.5
        y, x, _ = _make_synthetic_pair(n=1000, beta_true=beta_true, seed=42)
        hedge, _, _ = fit(y, x, delta=1e-4, vt=1e-3)

        final_beta = float(hedge.iloc[-1])
        assert abs(final_beta - beta_true) < 0.25, (
            f"final hedge ratio {final_beta:.3f} too far from β={beta_true}"
        )

    def test_converges_across_seeds(self):
        """Check convergence isn't a fluke of one particular random seed."""
        beta_true = 1.5
        for seed in [7, 13, 99, 2024]:
            y, x, _ = _make_synthetic_pair(n=800, beta_true=beta_true, seed=seed)
            hedge, _, _ = fit(y, x)
            # median of the last 20% — more stable than the final value
            late_median = float(np.median(hedge.iloc[int(0.8 * len(hedge)):].values))
            assert abs(late_median - beta_true) < 0.40, (
                f"seed {seed}: late-stage median β={late_median:.3f} vs true={beta_true}"
            )

    def test_positive_hedge_for_positive_relationship(self):
        y, x, _ = _make_synthetic_pair(n=600, beta_true=0.8, seed=5)
        hedge, _, _ = fit(y, x)
        assert float(hedge.iloc[-1]) > 0

    def test_output_length_matches_input(self):
        y, x, _ = _make_synthetic_pair(n=500)
        hedge, intercept, spread = fit(y, x)
        assert len(hedge) == len(intercept) == len(spread) == 500


class TestZScore:
    """After the warmup, z-score should be approximately N(0,1)."""

    def test_mean_near_zero_after_warmup(self):
        y, x, _ = _make_synthetic_pair(n=1000, seed=42)
        _, _, spread = fit(y, x)
        z = zscore(spread, window=20)
        z_late = z.dropna().iloc[60:]  # skip warmup
        assert abs(float(z_late.mean())) < 0.15, (
            f"z-score mean {z_late.mean():.4f} too far from zero — spread not centred"
        )

    def test_std_near_one_after_warmup(self):
        """By construction of z-score normalisation, std should be ≈ 1."""
        y, x, _ = _make_synthetic_pair(n=1000, seed=42)
        _, _, spread = fit(y, x)
        z = zscore(spread, window=20)
        z_late = z.dropna().iloc[60:]
        std_z = float(z_late.std(ddof=1))
        assert 0.7 < std_z < 1.3, f"z-score std {std_z:.4f} outside [0.7, 1.3]"

    def test_nan_for_initial_window_rows(self):
        """The first window−1 values should be NaN — not enough history yet."""
        y, x, _ = _make_synthetic_pair(n=200)
        _, _, spread = fit(y, x)
        z = zscore(spread, window=20)
        assert z.iloc[:19].isna().all(), "expected NaN for first window-1 z-score values"

    def test_longer_window_produces_more_nan(self):
        y, x, _ = _make_synthetic_pair(n=300)
        _, _, spread = fit(y, x)
        assert zscore(spread, window=10).notna().sum() > zscore(spread, window=40).notna().sum()


class TestKalmanEdgeCases:

    def test_raises_on_too_short_series(self):
        """< 4 observations should raise ValueError."""
        dates = _make_dates(3)
        y = pd.Series([1.0, 2.0, 3.0], index=dates)
        x = pd.Series([1.0, 2.0, 3.0], index=dates)
        with pytest.raises(ValueError, match="overlapping observations"):
            fit(y, x)

    def test_raises_on_non_overlapping_dates(self):
        y = pd.Series(np.random.randn(100), index=pd.bdate_range("2020-01-01", periods=100, freq="B"))
        x = pd.Series(np.random.randn(100), index=pd.bdate_range("2022-01-01", periods=100, freq="B"))
        with pytest.raises(ValueError):
            fit(y, x)

    def test_spread_variance_less_than_raw_y(self):
        """The Kalman spread should have lower variance than Y — that's the whole point."""
        y, x, _ = _make_synthetic_pair(n=600, beta_true=1.5)
        _, _, spread = fit(y, x)
        assert float(spread.var()) < float(y.var()), (
            f"spread var {spread.var():.4f} should be < raw Y var {y.var():.4f}"
        )
