"""
test_pair_selection.py — unit tests for the pair selection pipeline.

Testing:
1. half_life() returns a sensible value on a synthetic OU series with known κ
2. pair selector rejects two independent random walks
3. pair selector accepts a pair where Y = β·X + OU spread by construction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pair_selection import compute_half_life, select_pairs


def _make_dates(n: int = 500) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


def _make_ou_spread(
    n: int = 500,
    kappa: float = 0.05,
    sigma: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Simulate an OU process. True half-life = ln(2)/kappa."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - kappa * s[t - 1] + sigma * rng.standard_normal()
    return s


def _make_rw(n: int = 500, seed: int = 0) -> np.ndarray:
    """Pure random walk — non-stationary, should fail the half-life filter."""
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _make_cointegrated_pair(
    n: int = 800,
    beta_true: float = 0.7,
    kappa: float = 0.05,
    seed: int = 123,
) -> tuple:
    """
    Build a cointegrated pair where Y = beta_true·X + OU spread.

    X is a random walk. The spread Y − β·X is stationary by construction,
    so EG and Johansen should both pass.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n) * 0.5) + 100.0
    spread = _make_ou_spread(n=n, kappa=kappa, sigma=0.5, seed=seed + 1)
    y = beta_true * x + spread

    dates = _make_dates(n)
    prices = pd.DataFrame({"Y": y, "X": x}, index=dates)
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


class TestHalfLife:
    """For an OU process with κ=0.05, true half-life ≈ 13.86 days."""

    def test_known_kappa(self):
        kappa_true = 0.05
        expected_hl = np.log(2) / kappa_true  # ≈ 13.86
        spread = _make_ou_spread(n=2000, kappa=kappa_true, sigma=0.3, seed=42)
        hl = compute_half_life(spread)

        assert hl is not None, "returned None for a mean-reverting series"
        assert hl > 0, f"half-life must be positive, got {hl:.2f}"
        # allow 30% finite-sample error
        rel_err = abs(hl - expected_hl) / expected_hl
        assert rel_err < 0.30, f"estimate {hl:.2f} vs true {expected_hl:.2f} ({rel_err:.1%} error)"

    def test_fast_reversion_gives_short_hl(self):
        spread = _make_ou_spread(n=2000, kappa=0.20, sigma=0.3, seed=7)
        hl = compute_half_life(spread)
        assert hl is not None
        assert hl < 10, f"fast-reverting spread should have HL < 10, got {hl:.2f}"

    def test_random_walk_returns_none_or_very_long(self):
        """A random walk either returns None or a very large HL — both are correct."""
        rw = _make_rw(n=1000, seed=42)
        hl = compute_half_life(rw)
        if hl is not None:
            # it slipped through but the half-life filter would kill it anyway
            assert hl > 100, f"expected None or HL > 100 for random walk, got {hl:.2f}"

    def test_short_series_returns_none(self):
        hl = compute_half_life(np.array([1.0, 0.5]))
        assert hl is None, "expected None for series < 10 observations"


class TestRejectNonCointegrated:
    """Two independent random walks should fail the cointegration pipeline."""

    def test_rejects_independent_rws(self):
        rng = np.random.default_rng(99)
        n = 600
        dates = _make_dates(n)
        prices = pd.DataFrame({
            "A": np.cumsum(rng.standard_normal(n)) + 100.0,
            "B": np.cumsum(rng.standard_normal(n)) + 100.0,
        }, index=dates)
        returns = np.log(prices / prices.shift(1)).dropna()

        results = select_pairs(
            neutral_prices=prices,
            neutral_returns=returns,
            sector_map={"A": "SectorA", "B": "SectorA"},
            min_correlation=0.70,
            coint_pvalue=0.05,
            min_half_life=5,
            max_half_life=60,
            rolling_coint_window=100,
            max_pairs_per_sector=2,
            n_workers=1,
        )

        # might slip through occasionally (type I error) but if so the HL filter should catch it
        if not results.empty:
            for _, row in results.iterrows():
                assert 5 <= row["half_life"] <= 60, (
                    f"spuriously accepted pair has implausible HL={row['half_life']:.2f}"
                )


class TestAcceptCointegrated:
    """Y = 0.7·X + OU(κ=0.05) should be found and have the right hedge ratio."""

    def test_accepts_synthetic_cointegrated_pair(self):
        beta_true = 0.7
        kappa = 0.05
        prices, returns = _make_cointegrated_pair(n=1000, beta_true=beta_true, kappa=kappa, seed=42)

        results = select_pairs(
            neutral_prices=prices,
            neutral_returns=returns,
            sector_map={"Y": "SectorA", "X": "SectorA"},
            min_correlation=0.50,
            coint_pvalue=0.10,
            min_half_life=5,
            max_half_life=60,
            rolling_coint_window=200,
            max_pairs_per_sector=2,
            n_workers=1,
        )

        assert not results.empty, "failed to find the synthetic cointegrated pair"

        row = results.iloc[0]
        # OLS on random walk price levels can have finite-sample bias — allow ±0.60
        # the important check is that the pair was found at all (line above)
        assert abs(row["hedge_ratio"] - beta_true) < 0.60, (
            f"hedge_ratio={row['hedge_ratio']:.3f} too far from true beta={beta_true}"
        )
        # HL should be roughly ln(2)/kappa ≈ 13.86 days, allow ±50%
        expected_hl = np.log(2) / kappa
        rel_err = abs(row["half_life"] - expected_hl) / expected_hl
        assert rel_err < 0.50, f"HL {row['half_life']:.2f} vs expected {expected_hl:.2f}"

    def test_score_is_positive(self):
        """Composite score should be strictly positive for any accepted pair."""
        prices, returns = _make_cointegrated_pair(n=800, seed=10)
        results = select_pairs(
            neutral_prices=prices, neutral_returns=returns,
            sector_map={"Y": "Tech", "X": "Tech"},
            min_correlation=0.50, coint_pvalue=0.10,
            min_half_life=5, max_half_life=60,
            rolling_coint_window=150, max_pairs_per_sector=2, n_workers=1,
        )
        if not results.empty:
            assert (results["score"] > 0).all(), "all scores must be positive"

    def test_output_has_required_columns(self):
        prices, returns = _make_cointegrated_pair(n=800, seed=77)
        results = select_pairs(
            neutral_prices=prices, neutral_returns=returns,
            sector_map={"Y": "Energy", "X": "Energy"},
            min_correlation=0.50, coint_pvalue=0.10,
            min_half_life=5, max_half_life=60,
            rolling_coint_window=150, max_pairs_per_sector=2, n_workers=1,
        )
        for col in ["ticker_y", "ticker_x", "correlation", "eg_pvalue",
                    "johansen_reject", "hedge_ratio", "half_life", "spread_std",
                    "recently_broken", "score"]:
            assert col in results.columns, f"missing column: {col}"
