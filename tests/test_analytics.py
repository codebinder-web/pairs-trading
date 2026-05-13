"""
test_analytics.py — unit tests for the analytics module.

All metrics are from scratch so we test them against known ground-truth values.
These are the kind of tests that catch the embarrassing mistakes — wrong sign,
missing annualisation factor, off-by-one in the drawdown calculation.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    cagr_pct,
    calmar_ratio,
    conditional_value_at_risk,
    excess_kurtosis,
    max_drawdown_pct,
    monthly_returns_table,
    omega_ratio,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    total_return_pct,
    trade_metrics,
    value_at_risk,
)
from src.backtest import TradeRecord


def _flat_returns(n: int = 252) -> pd.Series:
    return pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=n, freq="B"))


def _constant_growth_equity(start: float, end: float, n: int = 252) -> pd.Series:
    """Equity curve growing at a constant daily rate from start to end."""
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    daily_r = (end / start) ** (1.0 / n) - 1.0
    return pd.Series(start * (1.0 + daily_r) ** np.arange(n), index=dates)


def _known_drawdown_equity() -> pd.Series:
    """100 → 200 → 100 → 150. Max DD = -50% from peak of 200."""
    dates = pd.bdate_range("2020-01-01", periods=4, freq="B")
    return pd.Series([100.0, 200.0, 100.0, 150.0], index=dates)


def _make_trades(wins: List[float], losses: List[float], n_stops: int = 0) -> List[TradeRecord]:
    """Build a synthetic trade log from win/loss P&L lists."""
    trades = []
    entry = pd.Timestamp("2020-01-02")
    for pnl in wins:
        exit_ = entry + pd.Timedelta(days=5)
        trades.append(TradeRecord(
            entry_date=entry, exit_date=exit_,
            direction=1, pnl_pct=pnl, duration_days=5,
            entry_z=-2.1, exit_z=-0.2, stop_triggered=False,
        ))
        entry = exit_ + pd.Timedelta(days=1)
    for i, pnl in enumerate(losses):
        exit_ = entry + pd.Timedelta(days=5)
        trades.append(TradeRecord(
            entry_date=entry, exit_date=exit_,
            direction=1, pnl_pct=pnl, duration_days=5,
            entry_z=-2.1, exit_z=4.2, stop_triggered=(i < n_stops),
        ))
        entry = exit_ + pd.Timedelta(days=1)
    return trades


class TestSharpe:
    def test_zero_on_flat_returns(self):
        assert sharpe_ratio(_flat_returns()) == 0.0

    def test_positive_for_positive_mean(self):
        r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 252))
        assert sharpe_ratio(r) > 0

    def test_annualisation_applied(self):
        """Very consistent small positive return should give Sharpe >> 1."""
        r = pd.Series([0.001] * 252 + [0.0009] * 252)
        assert sharpe_ratio(r) > 10

    def test_symmetric_oscillation_gives_near_zero(self):
        r = pd.Series([0.01, -0.01] * 126)
        assert abs(sharpe_ratio(r)) < 0.1


class TestMaxDrawdown:
    def test_known_50_pct_drawdown(self):
        mdd = max_drawdown_pct(_known_drawdown_equity())
        assert abs(mdd - (-50.0)) < 0.01, f"expected -50.0%, got {mdd:.4f}%"

    def test_monotone_equity_zero_drawdown(self):
        assert max_drawdown_pct(_constant_growth_equity(100, 200)) == 0.0

    def test_max_dd_always_non_positive(self):
        eq = pd.Series(np.cumprod(1.0 + np.random.default_rng(42).normal(0, 0.01, 500)))
        assert max_drawdown_pct(eq) <= 0.0


class TestCAGR:
    def test_doubling_in_two_years(self):
        """equity 100 → 200 in ~2 years → CAGR ≈ 41.4%"""
        eq = _constant_growth_equity(100.0, 200.0, n=504)
        assert abs(cagr_pct(eq) - 41.42) < 2.0

    def test_flat_equity_gives_zero(self):
        dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
        eq = pd.Series(1.0, index=dates)
        assert abs(cagr_pct(eq)) < 0.01

    def test_tripling_in_one_year(self):
        """equity 100 → 300 in ~1 year → CAGR ≈ 200%"""
        assert abs(cagr_pct(_constant_growth_equity(100.0, 300.0, n=252)) - 200.0) < 5.0

    def test_calmar_is_cagr_over_maxdd(self):
        """Calmar = CAGR / |MaxDD| — verify internally consistent."""
        dates = pd.bdate_range("2020-01-01", periods=252, freq="B")
        values = np.linspace(100, 150, 252)
        values[126] = 200.0
        values[127:189] = np.linspace(120, 150, 62)
        eq = pd.Series(values, index=dates)
        ret = eq.pct_change().dropna()

        c = cagr_pct(eq)
        mdd = max_drawdown_pct(eq)
        calmar = calmar_ratio(ret, eq)
        if abs(mdd) > 0.001:
            expected = c / abs(mdd)
            assert abs(calmar - expected) < 0.001


class TestTradeMetrics:
    def test_win_rate_3_wins_2_losses(self):
        """3 wins, 2 losses → win_rate = 0.60"""
        m = trade_metrics(_make_trades(wins=[1.0, 2.0, 0.5], losses=[-0.5, -1.0]))
        assert abs(m["win_rate"] - 0.60) < 1e-9

    def test_profit_factor(self):
        """wins sum = 3.5, loss sum = 1.5 → profit_factor = 7/3 ≈ 2.333"""
        m = trade_metrics(_make_trades(wins=[1.0, 2.0, 0.5], losses=[-0.5, -1.0]))
        assert abs(m["profit_factor"] - 3.5 / 1.5) < 1e-6

    def test_expectancy(self):
        """expectancy = mean(all pnls) = (3.5 - 1.5) / 5 = 0.4"""
        m = trade_metrics(_make_trades(wins=[1.0, 2.0, 0.5], losses=[-0.5, -1.0]))
        assert abs(m["expectancy_pct"] - 0.4) < 1e-6

    def test_stop_rate(self):
        """2 stops out of 5 trades → stop_rate = 0.40"""
        m = trade_metrics(_make_trades(wins=[1.0, 2.0], losses=[-1.0, -2.0, -3.0], n_stops=2))
        assert abs(m["stop_rate"] - 0.40) < 1e-9

    def test_empty_trade_log(self):
        m = trade_metrics([])
        assert m["n_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_all_wins_gives_inf_profit_factor(self):
        m = trade_metrics(_make_trades(wins=[1.0, 0.5, 2.0], losses=[]))
        assert m["win_rate"] == 1.0
        assert math.isinf(m["profit_factor"]) or m["profit_factor"] > 1000


class TestMetricEdgeCases:
    def test_sortino_zero_mean(self):
        assert abs(sortino_ratio(pd.Series([0.0] * 252))) < 0.01

    def test_omega_symmetric_returns_near_one(self):
        o = omega_ratio(pd.Series([0.01, -0.01] * 126), threshold=0.0)
        assert abs(o - 1.0) < 0.01

    def test_var_negative_for_volatile_series(self):
        r = pd.Series(np.random.default_rng(7).normal(0.0, 0.02, 1000))
        assert value_at_risk(r, 0.95) > 0  # VaR is returned as a positive loss

    def test_cvar_geq_var(self):
        """CVaR is always >= VaR (expected shortfall is worse than the cutoff)."""
        r = pd.Series(np.random.default_rng(42).normal(0.0, 0.01, 500))
        var = value_at_risk(r, 0.95)
        cvar = conditional_value_at_risk(r, 0.95)
        assert cvar >= var - 1e-9

    def test_total_return_100_to_150(self):
        eq = pd.Series(np.linspace(100, 150, 252),
                       index=pd.bdate_range("2020-01-01", periods=252, freq="B"))
        assert abs(total_return_pct(eq) - 50.0) < 0.1


class TestMonthlyReturnsTable:
    def test_shape_for_two_years(self):
        dates = pd.date_range("2020-01-02", "2021-12-31", freq="B")
        eq = pd.Series(
            np.cumprod(1.0 + np.random.default_rng(0).normal(0, 0.005, len(dates))),
            index=dates,
        )
        tbl = monthly_returns_table(eq)
        assert tbl.shape[0] == 2, f"expected 2 year rows, got {tbl.shape[0]}"

    def test_empty_for_short_equity(self):
        eq = pd.Series([1.0, 1.01, 1.02],
                       index=pd.bdate_range("2020-01-01", periods=3, freq="B"))
        assert monthly_returns_table(eq).empty
