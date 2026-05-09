"""
sensitivity_analysis.py — 2D parameter grid search over entry_z × max_half_life.

The whole point is to see whether the strategy's results are robust or whether
they depend on having picked the exact right thresholds. If Sharpe is high in
a 4×4 grid of parameters, that's evidence of genuine edge. If it's only good
in one cell, it's probably overfit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Results for one cell of the parameter grid."""
    entry_z: float
    max_half_life: int
    sharpe_ratio: float
    cagr_pct: float
    max_drawdown_pct: float
    n_pairs: int


def run_sensitivity(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    base_config: Config,
    entry_z_values: Optional[List[float]] = None,
    max_hl_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Sweep over entry_z × max_half_life and compute Sharpe for each combination.

    Returns a DataFrame with columns: entry_z, max_half_life, sharpe_ratio,
    cagr_pct, max_drawdown_pct, n_pairs. Useful as input to visualisation.sensitivity_heatmap.
    """
    from src import analytics, backtest, data_loader, factor_neutralisation
    from src import kalman_filter, ou_process, pair_selection, regime_detector, strategy

    entry_z_values = entry_z_values or base_config.sensitivity_entry_z
    max_hl_values = max_hl_values or base_config.sensitivity_max_hl

    n_combos = len(entry_z_values) * len(max_hl_values)
    logger.info("sensitivity sweep: %d combinations (%d entry_z × %d max_hl)",
                n_combos, len(entry_z_values), len(max_hl_values))

    # precompute neutralised prices once — same for all grid cells
    spy_returns = data_loader.log_returns(spy_prices.to_frame()).iloc[:, 0]
    all_returns = data_loader.log_returns(prices)
    neutral_prices, neutral_returns, _ = factor_neutralisation.neutralise(
        prices=prices,
        returns=all_returns,
        market_returns=spy_returns,
        sector_map=base_config.sector_map,
        neutralise_market=base_config.neutralise_market,
        neutralise_sector=base_config.neutralise_sector,
    )

    # fit regime detector once on full history (this is in-sample sensitivity, not walk-forward)
    reg_detector = regime_detector.fit(
        market_returns=spy_returns,
        vix_series=vix_prices,
        vix_threshold=base_config.vix_threshold,
        hmm_n_states=base_config.hmm_n_states,
        use_hmm=base_config.use_hmm,
    )
    regime = reg_detector.predict(prices.index)

    results = []
    for entry_z in entry_z_values:
        for max_hl in max_hl_values:
            try:
                pairs_df = pair_selection.select_pairs(
                    neutral_prices=neutral_prices,
                    neutral_returns=neutral_returns,
                    sector_map=base_config.sector_map,
                    min_correlation=base_config.min_correlation,
                    coint_pvalue=base_config.coint_pvalue,
                    min_half_life=base_config.min_half_life,
                    max_half_life=max_hl,
                    rolling_coint_window=base_config.rolling_coint_window,
                    max_pairs_per_sector=base_config.max_pairs_per_sector,
                )

                if pairs_df.empty:
                    results.append(SensitivityResult(
                        entry_z=entry_z, max_half_life=max_hl,
                        sharpe_ratio=0.0, cagr_pct=0.0, max_drawdown_pct=0.0, n_pairs=0,
                    ))
                    continue

                pair_results = []
                for _, row in pairs_df.head(base_config.top_n_pairs).iterrows():
                    ticker_y = str(row["ticker_y"])
                    ticker_x = str(row["ticker_x"])
                    if ticker_y not in neutral_prices.columns or ticker_x not in neutral_prices.columns:
                        continue

                    hedge_ratio, intercept, spread = kalman_filter.fit(
                        y=neutral_prices[ticker_y],
                        x=neutral_prices[ticker_x],
                        delta=base_config.kalman_delta,
                        vt=base_config.kalman_vt,
                    )

                    z = kalman_filter.zscore(spread, window=20)

                    try:
                        ou_params = ou_process.fit(spread, method=base_config.ou_fitting_method)
                    except Exception:
                        ou_params = None

                    sigs = strategy.generate_signals(
                        zscore=z,
                        regime=regime,
                        entry_z=entry_z,
                        exit_z=base_config.exit_z,
                        stop_z=base_config.stop_z,
                        ou_params=ou_params,
                        use_ou_thresholds=False,  # fix thresholds for clean grid sweep
                        sharpe_target=base_config.sharpe_target,
                        spread=spread,
                    )

                    ry = data_loader.log_returns(prices[[ticker_y]]).iloc[:, 0]
                    rx = data_loader.log_returns(prices[[ticker_x]]).iloc[:, 0]

                    pr = backtest.run_pair_backtest(
                        ticker_y=ticker_y, ticker_x=ticker_x,
                        returns_y=ry, returns_x=rx,
                        signals=sigs, regime=regime,
                        hedge_ratio=hedge_ratio, spread=spread, zscore=z,
                        transaction_cost=base_config.transaction_cost,
                        ou_params=ou_params,
                    )
                    pair_results.append(pr)

                portfolio = backtest.run_portfolio_backtest(
                    pair_results=pair_results,
                    spy_returns=spy_returns,
                    initial_capital=base_config.initial_capital,
                    sizing_method=base_config.sizing_method,
                    max_pair_weight=base_config.max_pair_weight,
                )

                sharpe = analytics.sharpe_ratio(portfolio.returns)
                cagr = analytics.cagr_pct(portfolio.equity_curve)
                mdd = analytics.max_drawdown_pct(portfolio.equity_curve)

                results.append(SensitivityResult(
                    entry_z=entry_z, max_half_life=max_hl,
                    sharpe_ratio=sharpe, cagr_pct=cagr,
                    max_drawdown_pct=mdd, n_pairs=len(pair_results),
                ))
                logger.info("entry_z=%.1f max_hl=%d → sharpe=%.3f", entry_z, max_hl, sharpe)

            except Exception as exc:
                logger.error("grid cell (%.1f, %d) failed: %s", entry_z, max_hl, exc)
                results.append(SensitivityResult(
                    entry_z=entry_z, max_half_life=max_hl,
                    sharpe_ratio=float("nan"), cagr_pct=float("nan"),
                    max_drawdown_pct=float("nan"), n_pairs=0,
                ))

    records = [
        {
            "entry_z": r.entry_z,
            "max_half_life": r.max_half_life,
            "sharpe_ratio": r.sharpe_ratio,
            "cagr_pct": r.cagr_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "n_pairs": r.n_pairs,
        }
        for r in results
    ]
    return pd.DataFrame(records)
