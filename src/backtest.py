"""
backtest.py — vectorised single-pair and portfolio backtester.

Key design decisions:
- Signals are lagged by 1 day before execution. Signal at close of day t
  means trade executes at open of day t+1. This is enforced via np.roll(sig, 1)
  and setting pos[0]=0. No lookahead.
- Dollar-neutral: long Y for $W, short β·X for $W. Net market exposure should
  be close to zero if factor neutralisation did its job.
- Transaction costs on position changes — charged on both legs, round-trip = 2·tc.
  Spread std keeps blowing up on some pairs without this being applied correctly,
  so be careful if you change the cost structure.
- Kelly sizing uses in-sample pair return stats. Half-Kelly is the default
  because full Kelly drawdowns are brutal in practice even when the math is right.

TODO: add slippage model at some point — currently assumes we always trade at close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ou_process import OUParams

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """One completed round-trip trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int           # +1 = long Y / short X, -1 = short Y / long X
    pnl_pct: float          # net P&L as % of notional
    duration_days: int
    entry_z: float
    exit_z: float
    stop_triggered: bool


@dataclass
class PairResult:
    """Full results for a single pair backtest."""
    ticker_y: str
    ticker_x: str
    hedge_ratio: pd.Series
    spread: pd.Series
    zscore: pd.Series
    signals: pd.Series
    regime: pd.Series
    returns_gross: pd.Series
    returns_net: pd.Series
    equity_curve: pd.Series
    trades: List[TradeRecord]
    ou_params: Optional[OUParams]
    sizing_weight: float = 1.0


@dataclass
class PortfolioResult:
    """Aggregated portfolio results."""
    equity_curve: pd.Series
    returns: pd.Series
    pair_results: List[PairResult]
    portfolio_beta: float
    portfolio_correlation_to_spy: float
    initial_capital: float = 100_000.0


def run_pair_backtest(
    ticker_y: str,
    ticker_x: str,
    returns_y: pd.Series,
    returns_x: pd.Series,
    signals: pd.Series,
    regime: pd.Series,
    hedge_ratio: pd.Series,
    spread: pd.Series,
    zscore: pd.Series,
    transaction_cost: float = 0.001,
    ou_params: Optional[OUParams] = None,
    sizing_weight: float = 1.0,
) -> PairResult:
    """
    Vectorised backtest for a single pair.

    Lag signals by 1 day so we're not trading on prices we haven't seen yet.
    Gross P&L = position × (r_Y − β_t · r_X) where β_t is time-varying.
    Transaction cost is charged on position changes — 2×tc per unit for both legs.
    """
    common = (
        returns_y.index
        .intersection(returns_x.index)
        .intersection(signals.index)
        .intersection(hedge_ratio.index)
    )
    if len(common) < 5:
        logger.warning("pair %s/%s: only %d overlapping dates — skipping", ticker_y, ticker_x, len(common))
        empty = pd.Series(dtype=float)
        return PairResult(
            ticker_y=ticker_y, ticker_x=ticker_x,
            hedge_ratio=empty, spread=empty, zscore=empty,
            signals=empty, regime=empty,
            returns_gross=empty, returns_net=empty,
            equity_curve=pd.Series([1.0], dtype=float),
            trades=[], ou_params=ou_params, sizing_weight=sizing_weight,
        )

    ry = returns_y.loc[common].values.astype(float)
    rx = returns_x.loc[common].values.astype(float)
    sig_raw = signals.reindex(common, fill_value=0).values.astype(int)
    beta = hedge_ratio.reindex(common, method="ffill").values.astype(float)

    # lag by one day so we're not trading on prices we haven't seen yet
    pos = np.roll(sig_raw, 1)
    pos[0] = 0

    # gross return = position × (r_Y − β_t · r_X)
    spread_return = ry - beta * rx
    gross_return = pos.astype(float) * spread_return

    # transaction cost on position changes — both legs, round-trip = 2×tc
    cost = np.abs(np.diff(pos.astype(float), prepend=0.0)) * 2.0 * transaction_cost
    net_return = gross_return - cost

    equity = np.cumprod(1.0 + net_return)

    z_arr = zscore.reindex(common, method="ffill").values.astype(float)
    trades = _build_trade_log(common, pos, net_return, z_arr)

    logger.debug(
        "pair %s/%s: %d trades, final equity=%.4f",
        ticker_y, ticker_x, len(trades), equity[-1] if len(equity) else 1.0,
    )

    return PairResult(
        ticker_y=ticker_y,
        ticker_x=ticker_x,
        hedge_ratio=hedge_ratio.reindex(common),
        spread=spread.reindex(common),
        zscore=zscore.reindex(common),
        signals=signals.reindex(common, fill_value=0),
        regime=regime.reindex(common, fill_value=0),
        returns_gross=pd.Series(gross_return, index=common, name="gross_return"),
        returns_net=pd.Series(net_return, index=common, name="net_return"),
        equity_curve=pd.Series(equity, index=common, name="equity"),
        trades=trades,
        ou_params=ou_params,
        sizing_weight=sizing_weight,
    )


def run_portfolio_backtest(
    pair_results: List[PairResult],
    spy_returns: pd.Series,
    initial_capital: float = 100_000.0,
    sizing_method: str = "half_kelly",
    max_pair_weight: float = 0.15,
) -> PortfolioResult:
    """
    Combine individual pair results into a weighted portfolio equity curve.

    Weights are normalised to sum to ≤ 1.0, so the portfolio may hold
    some cash when Kelly sizing constraints bite.
    """
    if not pair_results:
        logger.warning("no pair results to combine")
        empty = pd.Series([1.0], dtype=float, name="portfolio_equity")
        return PortfolioResult(
            equity_curve=empty, returns=pd.Series([0.0], dtype=float),
            pair_results=[], portfolio_beta=0.0, portfolio_correlation_to_spy=0.0,
            initial_capital=initial_capital,
        )

    weights = _compute_weights(pair_results, sizing_method, max_pair_weight)

    for pr, w in zip(pair_results, weights):
        pr.sizing_weight = w

    # align all pair returns into a matrix and take weighted sum
    returns_matrix = pd.DataFrame(
        {f"{pr.ticker_y}_{pr.ticker_x}": pr.returns_net for pr in pair_results}
    ).fillna(0.0)

    portfolio_returns = returns_matrix.values @ np.array(weights)
    portfolio_returns_series = pd.Series(
        portfolio_returns, index=returns_matrix.index, name="portfolio_return"
    )

    equity = np.cumprod(1.0 + portfolio_returns)
    equity_series = pd.Series(equity, index=returns_matrix.index, name="portfolio_equity")

    spy_aligned = spy_returns.reindex(returns_matrix.index, fill_value=0.0)
    beta, corr = _market_relationship(portfolio_returns, spy_aligned.values)

    logger.info(
        "portfolio: %d pairs, beta=%.3f, corr_to_spy=%.3f",
        len(pair_results), beta, corr,
    )

    return PortfolioResult(
        equity_curve=equity_series,
        returns=portfolio_returns_series,
        pair_results=pair_results,
        portfolio_beta=beta,
        portfolio_correlation_to_spy=corr,
        initial_capital=initial_capital,
    )


def _compute_weights(
    pair_results: List[PairResult],
    sizing_method: str,
    max_pair_weight: float,
) -> List[float]:
    """
    Compute portfolio weights for each pair.

    equal:      1/n for each pair
    kelly:      μ/σ² — theoretically optimal but blows up in practice
    half_kelly: μ/(2σ²) — standard compromise, much safer drawdown profile

    All weights are capped at max_pair_weight and normalised to sum ≤ 1.
    """
    n = len(pair_results)

    if sizing_method == "equal":
        w = np.full(n, 1.0 / n)

    elif sizing_method in ("kelly", "half_kelly"):
        raw = []
        for pr in pair_results:
            ret = pr.returns_net.values
            if len(ret) < 10 or np.std(ret) < 1e-10:
                raw.append(0.01)
                continue
            mu_hat = float(np.mean(ret))
            sigma2_hat = float(np.var(ret, ddof=1))
            kelly_f = mu_hat / sigma2_hat if sigma2_hat > 1e-12 else 0.0
            if sizing_method == "half_kelly":
                kelly_f /= 2.0
            raw.append(max(kelly_f, 0.0))
        w = np.array(raw)

    else:
        raise ValueError(f"unknown sizing_method '{sizing_method}'")

    w = np.clip(w, 0.0, max_pair_weight)

    # if Kelly gave zero for everything (negative in-sample mean), fall back to equal
    # so the portfolio actually allocates capital rather than sitting flat
    if w.sum() < 1e-9:
        logger.warning("kelly weights summed to zero — falling back to equal weighting")
        w = np.full(n, 1.0 / n)
        w = np.clip(w, 0.0, max_pair_weight)

    if w.sum() > 1.0:
        w = w / w.sum()

    return w.tolist()


def _build_trade_log(
    dates: pd.DatetimeIndex,
    pos: np.ndarray,
    net_returns: np.ndarray,
    z_arr: np.ndarray,
) -> List[TradeRecord]:
    """
    Extract completed round-trip trades from the position array.

    A trade starts when position goes from 0 to ±1, ends when it goes back to 0.
    Trades still open at the end of the series are excluded — incomplete.
    Flips (long → short without going flat) are treated as close + open.
    """
    trades = []
    n = len(pos)
    in_trade = False
    entry_idx = 0
    direction = 0
    stop_threshold = 4.0

    for t in range(n):
        if not in_trade:
            if pos[t] != 0:
                in_trade = True
                entry_idx = t
                direction = int(pos[t])
        else:
            if pos[t] == 0 or t == n - 1:
                if pos[t] != 0:
                    break  # still open at end — skip
                exit_idx = t
                entry_z_val = float(z_arr[entry_idx]) if not np.isnan(z_arr[entry_idx]) else 0.0
                exit_z_val = float(z_arr[exit_idx]) if not np.isnan(z_arr[exit_idx]) else 0.0
                trades.append(TradeRecord(
                    entry_date=dates[entry_idx],
                    exit_date=dates[exit_idx],
                    direction=direction,
                    pnl_pct=float(np.sum(net_returns[entry_idx:exit_idx + 1]) * 100.0),
                    duration_days=exit_idx - entry_idx,
                    entry_z=entry_z_val,
                    exit_z=exit_z_val,
                    stop_triggered=abs(exit_z_val) >= stop_threshold * 0.9,
                ))
                in_trade = False

            elif pos[t] != direction:
                # flip — close current trade, open the opposite immediately
                exit_idx = t - 1
                entry_z_val = float(z_arr[entry_idx]) if not np.isnan(z_arr[entry_idx]) else 0.0
                exit_z_val = float(z_arr[t]) if not np.isnan(z_arr[t]) else 0.0
                trades.append(TradeRecord(
                    entry_date=dates[entry_idx],
                    exit_date=dates[exit_idx],
                    direction=direction,
                    pnl_pct=float(np.sum(net_returns[entry_idx:exit_idx + 1]) * 100.0),
                    duration_days=exit_idx - entry_idx,
                    entry_z=entry_z_val,
                    exit_z=exit_z_val,
                    stop_triggered=False,
                ))
                in_trade = True
                entry_idx = t
                direction = int(pos[t])

    return trades


def _market_relationship(
    portfolio_returns: np.ndarray,
    spy_returns: np.ndarray,
) -> Tuple[float, float]:
    """Returns (beta, correlation) of portfolio vs SPY."""
    if len(portfolio_returns) < 2:
        return 0.0, 0.0
    try:
        var_spy = float(np.var(spy_returns, ddof=1))
        if var_spy < 1e-14:
            return 0.0, 0.0
        cov = float(np.cov(portfolio_returns, spy_returns)[0, 1])
        beta = cov / var_spy
        std_port = float(np.std(portfolio_returns, ddof=1))
        std_spy = float(np.std(spy_returns, ddof=1))
        corr = cov / (std_port * std_spy) if std_port > 1e-14 and std_spy > 1e-14 else 0.0
        return float(beta), float(corr)
    except Exception:
        return 0.0, 0.0
