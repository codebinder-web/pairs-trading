"""
factor_neutralisation.py — strip out market and sector beta before cointegration testing.

This is the step most implementations skip, and it matters a lot. If you run
cointegration on raw prices you'll find pairs that look stable but are just two
stocks with similar SPY betas. The spread doesn't revert — they both just track
the market. Factor-neutralising first means we're testing whether the *idiosyncratic*
components of two stocks are cointegrated, which is the actual claim we want to make.

For each stock i we run:
    r_{i,t} = α_i + β_mkt,i · r_SPY,t + β_sec,i · r_sector,t + ε_{i,t}

The sector factor is the equal-weight average of the other stocks in the same
sector — leave-one-out so we're not including the stock in its own regressor.

Neutral prices are then reconstructed as:
    p_i,t^neutral = 100 · exp(cumsum(ε_{i,t}))

Everything downstream works on these neutral prices and returns.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


def neutralise(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_map: Dict[str, str],
    neutralise_market: bool = True,
    neutralise_sector: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Regress out market and sector beta from each stock's returns.

    Returns (neutral_prices, neutral_returns, betas_df) where neutral_prices
    are the OLS residuals exponentiated back into price space, rebased to 100.
    betas_df has beta_mkt, beta_sec, alpha, r_squared per ticker.
    """
    if not neutralise_market and not neutralise_sector:
        logger.warning("both neutralise flags are False — returning raw returns unchanged")
        neutral_returns = returns.copy()
        neutral_prices = _reconstruct_prices(neutral_returns)
        betas_df = pd.DataFrame(columns=["ticker", "beta_mkt", "beta_sec", "alpha", "r_squared"])
        return neutral_prices, neutral_returns, betas_df

    common_dates = returns.index.intersection(market_returns.index)
    if len(common_dates) < 60:
        raise ValueError(
            f"only {len(common_dates)} overlapping dates — need at least 60 for regression"
        )

    ret = returns.loc[common_dates].copy()
    mkt = market_returns.loc[common_dates].copy()

    # skip tickers that don't have a sector — can't compute leave-one-out factor
    valid_tickers = [t for t in ret.columns if t in sector_map]
    dropped = [t for t in ret.columns if t not in sector_map]
    if dropped:
        logger.warning("dropping %d ticker(s) missing from sector_map: %s", len(dropped), dropped)
    ret = ret[valid_tickers]

    # precompute the leave-one-out sector return for each ticker
    sector_returns: Dict[str, pd.Series] = _compute_sector_returns(ret, sector_map)

    records = []
    residuals: Dict[str, pd.Series] = {}

    for ticker in valid_tickers:
        try:
            resid, beta_mkt, beta_sec, alpha, r2 = _regress_one(
                ticker=ticker,
                stock_ret=ret[ticker],
                mkt_ret=mkt,
                sector_ret=sector_returns.get(ticker),
                neutralise_market=neutralise_market,
                neutralise_sector=neutralise_sector,
            )
            residuals[ticker] = resid
            records.append({
                "ticker": ticker,
                "beta_mkt": beta_mkt,
                "beta_sec": beta_sec,
                "alpha": alpha,
                "r_squared": r2,
            })
        except Exception as exc:
            logger.error("neutralisation failed for %s: %s — dropping ticker", ticker, exc)

    if not residuals:
        raise RuntimeError("factor neutralisation produced no valid residuals — check input data")

    neutral_returns = pd.DataFrame(residuals, index=common_dates)
    neutral_prices = _reconstruct_prices(neutral_returns)
    betas_df = pd.DataFrame(records).set_index("ticker")

    logger.info(
        "neutralisation done: %d tickers, mean b_mkt=%.3f, mean b_sec=%.3f, mean R2=%.3f",
        len(residuals),
        betas_df["beta_mkt"].mean(),
        betas_df["beta_sec"].mean(),
        betas_df["r_squared"].mean(),
    )

    return neutral_prices, neutral_returns, betas_df


def _compute_sector_returns(
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
) -> Dict[str, pd.Series]:
    """
    Build a leave-one-out sector return for each ticker.

    Excluding the stock from its own sector average matters most for sectors
    with few members — otherwise you get a mechanical negative residual.
    """
    sectors: Dict[str, list] = {}
    for ticker, sector in sector_map.items():
        if ticker in returns.columns:
            sectors.setdefault(sector, []).append(ticker)

    result: Dict[str, pd.Series] = {}
    for ticker in returns.columns:
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        # everyone in the same sector except this ticker
        peers = [t for t in sectors.get(sector, []) if t != ticker]
        if not peers:
            # only stock in its sector — zero out the sector factor
            result[ticker] = pd.Series(0.0, index=returns.index, name=f"sec_{sector}")
            logger.debug("%s is the only member of '%s', sector factor = 0", ticker, sector)
        else:
            result[ticker] = returns[peers].mean(axis=1).rename(f"sec_{sector}")

    return result


def _regress_one(
    ticker: str,
    stock_ret: pd.Series,
    mkt_ret: pd.Series,
    sector_ret: Optional[pd.Series],
    neutralise_market: bool,
    neutralise_sector: bool,
) -> Tuple[pd.Series, float, float, float, float]:
    """OLS regression for one ticker, returns (residuals, beta_mkt, beta_sec, alpha, r2)."""
    regressors = []
    if neutralise_market:
        regressors.append(mkt_ret.rename("mkt"))
    if neutralise_sector and sector_ret is not None:
        regressors.append(sector_ret.rename("sec"))

    if not regressors:
        return stock_ret.copy(), 0.0, 0.0, 0.0, 0.0

    X_df = pd.concat(regressors, axis=1).loc[stock_ret.index]
    y = stock_ret.copy()

    # mask out any rows with NaN in y or X before fitting
    valid = y.notna() & X_df.notna().all(axis=1)
    y_clean = y[valid]
    X_clean = add_constant(X_df[valid], has_constant="add")

    if len(y_clean) < 30:
        logger.warning("only %d observations for %s regression — results may be noisy", len(y_clean), ticker)

    model = OLS(y_clean, X_clean).fit()

    alpha = float(model.params.get("const", 0.0))
    beta_mkt = float(model.params.get("mkt", 0.0))
    beta_sec = float(model.params.get("sec", 0.0))
    r2 = float(model.rsquared)

    # put residuals back on the full index, then fill the few masked rows
    resid = pd.Series(np.nan, index=stock_ret.index, name=ticker)
    resid[valid] = model.resid.values

    n_nan = resid.isna().sum()
    if n_nan > 0:
        logger.debug("%s: %d NaN residuals after OLS — filling", ticker, n_nan)
        resid = resid.ffill().bfill()

    return resid, beta_mkt, beta_sec, alpha, r2


def _reconstruct_prices(neutral_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Turn residual returns back into a price series starting at 100.

    p_{i,t}^neutral = 100 * exp(cumsum(ε_{i,t}))
    """
    return 100.0 * np.exp(neutral_returns.cumsum())
