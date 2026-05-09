"""
analytics.py — performance metrics, all implemented from scratch.

Using libraries like pyfolio is tempting but they're opaque and occasionally
wrong. Building these by hand means you know exactly what you're computing.
All formulas are standard — nothing exotic here.

Annualisation assumes 252 trading days throughout unless otherwise noted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import PairResult, PortfolioResult, TradeRecord

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------- return metrics ----------

def total_return_pct(equity: pd.Series) -> float:
    """(final_value / initial_value - 1) × 100."""
    if len(equity) < 2:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)


def cagr_pct(equity: pd.Series) -> float:
    """Compound annual growth rate as a percent."""
    if len(equity) < 2:
        return 0.0
    n_years = len(equity) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float(((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0) * 100.0)


def annualised_volatility_pct(returns: pd.Series) -> float:
    """Annualised daily return std × 100."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)


# ---------- risk-adjusted ----------

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio using daily returns.

    (mean_daily - rf_daily) / std_daily × √252
    """
    if len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    std = float(excess.std(ddof=1))
    if std < 1e-14:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe but penalises only downside deviation. Better for skewed distributions."""
    if len(returns) < 2:
        return 0.0
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std < 1e-14:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """CAGR / |max_drawdown|. Measures return per unit of worst-case loss."""
    mdd = abs(max_drawdown_pct(equity))
    if mdd < 1e-9:
        return 0.0
    return cagr_pct(equity) / mdd


def max_drawdown_duration(equity: pd.Series) -> int:
    """Number of trading days spent below a prior peak (longest drawdown period)."""
    if len(equity) < 2:
        return 0
    in_drawdown = equity < equity.cummax()
    max_dur = 0
    current_dur = 0
    for v in in_drawdown:
        if v:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """E[max(r-threshold, 0)] / E[max(threshold-r, 0)] — should be > 1."""
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns <= threshold]).sum()
    if losses < 1e-14:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualised mean excess return vs benchmark / tracking error."""
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0
    excess = returns.loc[common] - benchmark_returns.loc[common]
    te = float(excess.std(ddof=1))
    if te < 1e-14:
        return 0.0
    return float(excess.mean() / te * np.sqrt(TRADING_DAYS_PER_YEAR))


# ---------- drawdown ----------

def max_drawdown_pct(equity: pd.Series) -> float:
    """Maximum peak-to-trough decline as a percent (negative number)."""
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100.0
    return float(drawdown.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Full drawdown series — useful for plotting."""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max * 100.0


def avg_drawdown_pct(equity: pd.Series) -> float:
    """Average drawdown when in drawdown. More stable metric than max drawdown."""
    dd = drawdown_series(equity)
    negative = dd[dd < 0]
    return float(negative.mean()) if len(negative) > 0 else 0.0


# ---------- tail risk ----------

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at the given confidence level.

    Returns the loss (positive number) that is exceeded only (1 - confidence)%
    of the time. 5th percentile for 95% VaR.
    """
    if len(returns) < 10:
        return 0.0
    return float(-np.percentile(returns.dropna().values, (1 - confidence) * 100))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall — mean of returns below the VaR threshold.

    Better than VaR for capturing the severity of tail losses, not just the cutoff.
    """
    if len(returns) < 10:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns < -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ---------- distribution ----------

def skewness(returns: pd.Series) -> float:
    """Sample skewness. Positive = right-skewed (we want this for long strategies)."""
    return float(returns.skew()) if len(returns) >= 3 else 0.0


def excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (Fisher definition, normal = 0). High = fat tails."""
    return float(returns.kurtosis()) if len(returns) >= 4 else 0.0


def hit_rate(returns: pd.Series) -> float:
    """Fraction of daily returns that are positive."""
    pos = (returns > 0).sum()
    return float(pos / len(returns)) if len(returns) > 0 else 0.0


# ---------- market exposure ----------

def portfolio_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """OLS beta vs benchmark. Should be close to zero for a market-neutral strategy."""
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0
    port = returns.loc[common].values
    bench = benchmark_returns.loc[common].values
    var_bench = float(np.var(bench, ddof=1))
    if var_bench < 1e-14:
        return 0.0
    return float(np.cov(port, bench)[0, 1] / var_bench)


def portfolio_correlation(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Pearson correlation vs benchmark."""
    common = returns.index.intersection(benchmark_returns.index)
    if len(common) < 2:
        return 0.0
    try:
        return float(pd.Series(returns.loc[common].values).corr(pd.Series(benchmark_returns.loc[common].values)))
    except Exception:
        return 0.0


# ---------- trade-level ----------

def trade_metrics(trades: List[TradeRecord]) -> Dict[str, float]:
    """
    Compute trade-level stats from a list of completed TradeRecord objects.

    Returns a dict with: n_trades, win_rate, avg_pnl_pct, profit_factor,
    avg_duration_days, stop_rate, expectancy_pct.
    """
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0,
            "profit_factor": 0.0, "avg_duration_days": 0.0,
            "stop_rate": 0.0, "expectancy_pct": 0.0,
        }

    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    stops = sum(1 for t in trades if t.stop_triggered)
    durations = np.array([t.duration_days for t in trades])

    profit_factor = (
        float(wins.sum() / abs(losses.sum()))
        if len(losses) > 0 and abs(losses.sum()) > 1e-12
        else float("inf")
    )

    return {
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)),
        "avg_pnl_pct": float(pnls.mean()),
        "profit_factor": profit_factor,
        "avg_duration_days": float(durations.mean()),
        "stop_rate": float(stops / len(trades)),
        "pct_stopped_out": float(stops / len(trades)),  # alias for backward compat
        "expectancy_pct": float(pnls.mean()),
    }


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    """
    Pivot table of monthly returns by year × month — the classic heatmap input.

    Columns are months 1–12, rows are years.
    """
    if len(equity) < 2:
        return pd.DataFrame()
    monthly = equity.resample("ME").last().pct_change().dropna()
    monthly.index = pd.to_datetime(monthly.index)
    df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values * 100.0,
    })
    table = df.pivot(index="year", columns="month", values="return")
    return table


# ---------- summary tearsheets ----------

def tearsheet(
    result: PortfolioResult,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict:
    """
    Full portfolio tearsheet — all metrics in one dict.

    Parameters
    ----------
    result : PortfolioResult
        From run_portfolio_backtest().
    benchmark_returns : pd.Series, optional
        SPY returns for beta/correlation. If None, those fields are 0.
    risk_free_rate : float
        Annual risk-free rate (e.g. 0.05 for 5%).
    """
    ret = result.returns
    eq = result.equity_curve

    metrics: Dict = {
        "total_return_pct": total_return_pct(eq),
        "cagr_pct": cagr_pct(eq),
        "annualised_vol_pct": annualised_volatility_pct(ret),
        "sharpe_ratio": sharpe_ratio(ret, risk_free_rate),
        "sortino_ratio": sortino_ratio(ret, risk_free_rate),
        "calmar_ratio": calmar_ratio(ret, eq),
        "omega_ratio": omega_ratio(ret),
        "max_drawdown_pct": max_drawdown_pct(eq),
        "avg_drawdown_pct": avg_drawdown_pct(eq),
        "var_95_pct": value_at_risk(ret, 0.95),
        "cvar_95_pct": conditional_value_at_risk(ret, 0.95),
        "skewness": skewness(ret),
        "excess_kurtosis": excess_kurtosis(ret),
        "hit_rate": hit_rate(ret),
        "portfolio_beta": result.portfolio_beta,
        "portfolio_corr_to_spy": result.portfolio_correlation_to_spy,
        "n_pairs": len(result.pair_results),
        "initial_capital": result.initial_capital,
        "final_capital": result.initial_capital * eq.iloc[-1] if len(eq) > 0 else result.initial_capital,
    }

    # aggregate trade stats across all pairs
    all_trades = [t for pr in result.pair_results for t in pr.trades]
    metrics["trade_stats"] = trade_metrics(all_trades)

    if benchmark_returns is not None:
        metrics["information_ratio"] = information_ratio(ret, benchmark_returns)
        metrics["beta"] = portfolio_beta(ret, benchmark_returns)
        metrics["correlation_to_benchmark"] = portfolio_correlation(ret, benchmark_returns)

    return metrics


def pair_tearsheet(
    pair_result: PairResult,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Single-pair tearsheet — useful for debugging individual pairs."""
    ret = pair_result.returns_net
    eq = pair_result.equity_curve

    metrics: Dict = {
        "ticker_y": pair_result.ticker_y,
        "ticker_x": pair_result.ticker_x,
        "total_return_pct": total_return_pct(eq),
        "cagr_pct": cagr_pct(eq),
        "sharpe_ratio": sharpe_ratio(ret, risk_free_rate),
        "sortino_ratio": sortino_ratio(ret, risk_free_rate),
        "max_drawdown_pct": max_drawdown_pct(eq),
        "var_95_pct": value_at_risk(ret, 0.95),
        "annualised_vol_pct": annualised_volatility_pct(ret),
        "sizing_weight": pair_result.sizing_weight,
    }
    metrics["trade_stats"] = trade_metrics(pair_result.trades)
    return metrics
