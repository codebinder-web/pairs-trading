"""
pair_selection.py — five-stage cointegration screening pipeline.

Stage 1: correlation screen — cheap filter, O(N^2) in one vectorised call
Stage 2: Engle-Granger cointegration test
Stage 3: Johansen trace test — runs second because EG alone gave too many false positives
Stage 4: half-life filter — spread needs to mean-revert within a tradeable window
Stage 5: rolling stability check — catches pairs whose relationship quietly fell apart
Stage 6: sector concentration cap — no more than max_pairs_per_sector from any one sector

The cointegration tests (stages 2-4) are dispatched to multiprocessing.Pool — with
40 tickers that's 780 candidate pairs after stage 1, and running them serially
takes forever. Parallelising drops it to a few seconds.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


def select_pairs(
    neutral_prices: pd.DataFrame,
    neutral_returns: pd.DataFrame,
    sector_map: Dict[str, str],
    min_correlation: float = 0.25,
    coint_pvalue: float = 0.10,
    min_half_life: int = 3,
    max_half_life: int = 90,
    rolling_coint_window: int = 126,
    max_pairs_per_sector: int = 2,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run the full selection pipeline and return a scored DataFrame of accepted pairs.

    Output columns: ticker_y, ticker_x, sector_y, sector_x, correlation,
    eg_pvalue, johansen_reject, hedge_ratio, half_life, spread_std,
    recently_broken, score. Returns empty DataFrame if nothing survives.
    """
    tickers = sorted(set(neutral_prices.columns) & set(neutral_returns.columns))
    if len(tickers) < 2:
        logger.warning("fewer than 2 tickers — nothing to select")
        return _empty_pair_df()

    n_candidates = len(tickers) * (len(tickers) - 1) // 2
    logger.info("pair selection: %d tickers, %d candidate pairs", len(tickers), n_candidates)

    # stage 1 — correlation screen, all 780 pairs in one corr() call
    # drop any column with zero variance before corr() to avoid the numpy divide warning
    valid_cols = [t for t in tickers if neutral_returns[t].std(ddof=1) > 1e-12]
    tickers = valid_cols
    corr_matrix = neutral_returns[tickers].corr()
    candidate_pairs = _correlation_screen(corr_matrix, min_correlation)
    logger.info("after correlation screen (>= %.2f): %d pairs", min_correlation, len(candidate_pairs))

    if not candidate_pairs:
        return _empty_pair_df()

    # stages 2+3+4 — cointegration + half-life, parallelised across pairs
    worker_fn = partial(
        _test_pair,
        neutral_prices=neutral_prices,
        neutral_returns=neutral_returns,
        coint_pvalue=coint_pvalue,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        corr_matrix=corr_matrix,
    )

    n_workers = n_workers or mp.cpu_count()
    # only bother with a pool if there's enough work to justify the overhead
    if n_workers > 1 and len(candidate_pairs) >= 10:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, candidate_pairs)
    else:
        results = [worker_fn(p) for p in candidate_pairs]

    passed = [r for r in results if r is not None]
    logger.info("after cointegration + half-life: %d pairs", len(passed))

    if not passed:
        return _empty_pair_df()

    pairs_df = pd.DataFrame(passed)

    # stage 5 — rolling stability: drop pairs whose relationship recently broke
    pairs_df = _stability_check(pairs_df, neutral_prices, rolling_coint_window)
    n_before = len(pairs_df)
    pairs_df = pairs_df[~pairs_df["recently_broken"]].reset_index(drop=True)
    logger.info("after stability check: dropped %d, %d remain", n_before - len(pairs_df), len(pairs_df))

    if pairs_df.empty:
        return _empty_pair_df()

    # score and sort before applying the sector cap
    pairs_df["score"] = _composite_score(pairs_df)
    pairs_df = pairs_df.sort_values("score", ascending=False).reset_index(drop=True)
    pairs_df["sector_y"] = pairs_df["ticker_y"].map(sector_map)
    pairs_df["sector_x"] = pairs_df["ticker_x"].map(sector_map)

    # stage 6 — sector concentration cap, applied greedily from best score down
    pairs_df = _sector_cap(pairs_df, max_pairs_per_sector)
    logger.info("found %d pairs after all filters", len(pairs_df))

    col_order = [
        "ticker_y", "ticker_x", "sector_y", "sector_x",
        "correlation", "eg_pvalue", "johansen_reject",
        "hedge_ratio", "half_life", "spread_std",
        "recently_broken", "score",
    ]
    return pairs_df[col_order].reset_index(drop=True)


def _correlation_screen(
    corr_matrix: pd.DataFrame,
    min_correlation: float,
) -> List[Tuple[str, str]]:
    """Upper-triangle scan of the correlation matrix — avoids testing both (A,B) and (B,A)."""
    tickers = corr_matrix.columns.tolist()
    pairs = []
    for i, j in combinations(range(len(tickers)), 2):
        if abs(corr_matrix.iloc[i, j]) >= min_correlation:
            pairs.append((tickers[i], tickers[j]))
    return pairs


def _test_pair(
    pair: Tuple[str, str],
    neutral_prices: pd.DataFrame,
    neutral_returns: pd.DataFrame,
    coint_pvalue: float,
    min_half_life: int,
    max_half_life: int,
    corr_matrix: pd.DataFrame,
) -> Optional[Dict]:
    """
    Worker function for one (y, x) pair — runs in a subprocess.

    Must be a module-level function for multiprocessing to pickle it.
    Returns a result dict if the pair passes all checks, None otherwise.
    """
    ticker_y, ticker_x = pair

    try:
        y = neutral_prices[ticker_y].dropna()
        x = neutral_prices[ticker_x].dropna()
        common = y.index.intersection(x.index)

        # need at least 6 months of overlapping data
        if len(common) < 126:
            return None

        y_arr = y.loc[common].values.astype(float)
        x_arr = x.loc[common].values.astype(float)

        # engle-granger first — fast and catches the obvious non-cointegrated pairs
        _, eg_pvalue, _ = coint(y_arr, x_arr, trend="c")
        if eg_pvalue >= coint_pvalue:
            return None

        # OLS hedge ratio from the EG regression: y = β·x + α + ε
        X_reg = add_constant(x_arr, has_constant="add")
        ols_result = OLS(y_arr, X_reg).fit()
        hedge_ratio = float(ols_result.params[1])
        spread_arr = y_arr - hedge_ratio * x_arr

        # johansen confirms — EG alone gave too many false positives
        if not _johansen_test(y_arr, x_arr):
            return None

        # half-life from AR(1) on the spread: Δs_t = λ·s_{t-1} + ε
        hl = _half_life(spread_arr)
        if hl is None or hl < min_half_life or hl > max_half_life:
            return None

        return {
            "ticker_y": ticker_y,
            "ticker_x": ticker_x,
            "correlation": float(corr_matrix.loc[ticker_y, ticker_x]),
            "eg_pvalue": float(eg_pvalue),
            "johansen_reject": True,
            "hedge_ratio": hedge_ratio,
            "half_life": hl,
            "spread_std": float(np.std(spread_arr, ddof=1)),
            "recently_broken": False,
        }

    except Exception as exc:
        logger.debug("pair (%s, %s) failed: %s", ticker_y, ticker_x, exc)
        return None


def _johansen_test(y_arr: np.ndarray, x_arr: np.ndarray) -> bool:
    """
    Johansen trace test — reject r=0 at 5% means at least one cointegrating vector exists.

    statsmodels returns cvt with columns [10%, 5%, 1%], so index 1 is the 5% critical value.
    """
    try:
        data = np.column_stack([y_arr, x_arr])
        result = coint_johansen(data, det_order=0, k_ar_diff=1)
        return bool(result.lr1[0] > result.cvt[0, 1])
    except Exception as exc:
        logger.debug("johansen failed: %s", exc)
        return False


def _half_life(spread: np.ndarray) -> Optional[float]:
    """
    Estimate mean-reversion half-life from an AR(1) fit on the spread.

    AR(1): Δspread_t = λ · spread_{t-1} + ε_t
    half_life = −ln(2) / λ

    Returns None if λ >= 0 (spread is drifting, not reverting).
    """
    if len(spread) < 10:
        return None
    try:
        delta = np.diff(spread)
        lag = (spread[:-1] - spread[:-1].mean()).reshape(-1, 1)
        lam = float(OLS(delta, lag).fit().params[0])
        if lam >= 0:
            return None  # not mean-reverting
        return float(-np.log(2) / lam)
    except Exception:
        return None


def _stability_check(
    pairs_df: pd.DataFrame,
    neutral_prices: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Re-run EG on the most recent *window* days for each surviving pair.

    If the p-value is > 0.10 in the recent window, the relationship has probably
    broken. Using a slightly looser threshold (0.10) than the selection stage so
    we don't kick out pairs that are just going through a noisy period.
    """
    # looser threshold than the selection stage because cointegration tests have
    # lower power in a short (126-day) window — 0.10 would kill too many real pairs
    recent_break_threshold = 0.25

    def _check(row: pd.Series) -> bool:
        try:
            y = neutral_prices[row["ticker_y"]].dropna()
            x = neutral_prices[row["ticker_x"]].dropna()
            common = y.index.intersection(x.index)
            # if we have less data than the window just use all of it
            recent = common[-window:] if len(common) >= window else common
            _, p_val, _ = coint(
                y.loc[recent].values.astype(float),
                x.loc[recent].values.astype(float),
                trend="c",
            )
            return bool(p_val > recent_break_threshold)
        except Exception as exc:
            logger.debug("stability check failed for (%s, %s): %s", row["ticker_y"], row["ticker_x"], exc)
            # if we can't test it, assume it's broken rather than silently pass it through
            return True

    pairs_df = pairs_df.copy()
    pairs_df["recently_broken"] = pairs_df.apply(_check, axis=1)
    return pairs_df


def _composite_score(pairs_df: pd.DataFrame) -> pd.Series:
    """
    score = (1/half_life) × (1 - eg_pvalue) × |correlation| / spread_std

    Rewards fast reversion, statistical confidence, tight spread, strong correlation.
    """
    eps = 1e-9
    return (
        (1.0 / (pairs_df["half_life"] + eps))
        * (1.0 - pairs_df["eg_pvalue"])
        * pairs_df["correlation"].abs()
        / (pairs_df["spread_std"] + eps)
    )


def _sector_cap(pairs_df: pd.DataFrame, max_pairs_per_sector: int) -> pd.DataFrame:
    """
    Walk down the ranked list and greedily accept pairs until any sector hits its cap.

    Both ticker_y's sector and ticker_x's sector count — a pair between two
    energy stocks counts as +1 for Energy from both sides.
    """
    sector_counts: Dict[str, int] = {}
    accepted = []

    for idx, row in pairs_df.iterrows():
        sy = row.get("sector_y", "")
        sx = row.get("sector_x", "")
        if sector_counts.get(sy, 0) < max_pairs_per_sector and sector_counts.get(sx, 0) < max_pairs_per_sector:
            accepted.append(idx)
            sector_counts[sy] = sector_counts.get(sy, 0) + 1
            sector_counts[sx] = sector_counts.get(sx, 0) + 1

    return pairs_df.loc[accepted].reset_index(drop=True)


def _empty_pair_df() -> pd.DataFrame:
    """Empty DataFrame with the right columns so callers don't have to check types."""
    return pd.DataFrame(columns=[
        "ticker_y", "ticker_x", "sector_y", "sector_x",
        "correlation", "eg_pvalue", "johansen_reject",
        "hedge_ratio", "half_life", "spread_std",
        "recently_broken", "score",
    ])


def compute_half_life(spread: np.ndarray) -> Optional[float]:
    """Estimate mean reversion half-life from AR(1) fit on the spread. Public wrapper for tests."""
    return _half_life(spread)
