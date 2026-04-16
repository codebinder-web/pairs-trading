"""
data_loader.py — price download with disk caching and log-return computation.

Caching is MD5-keyed on sorted tickers + date range so the same request
always hits the same file. Running the full pipeline twice takes seconds
instead of minutes once the cache is warm.
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _cache_key(tickers: List[str], start: str, end: str) -> str:
    """MD5 of sorted tickers + dates — same request always maps to the same file."""
    payload = "|".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.parquet")


def load(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str = ".cache",
    min_coverage: float = 0.90,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers, with Parquet disk caching.

    Drops any ticker with less than min_coverage non-NaN rows — better to lose
    a ticker than carry a half-empty column through the whole pipeline.
    Remaining gaps get forward-filled then back-filled.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(tickers, start, end)
    path = _cache_path(cache_dir, key)

    if os.path.exists(path):
        logger.info("cache hit — loading from %s", path)
        prices = pd.read_parquet(path)
        logger.info("loaded %d tickers × %d days from cache", prices.shape[1], prices.shape[0])
        return prices

    logger.info("downloading %d tickers %s → %s", len(tickers), start, end)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
    except Exception as exc:
        raise RuntimeError(f"yfinance download failed: {exc}") from exc

    # auto_adjust=True makes Close the adjusted close, which is what we want
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()

    if prices.empty:
        raise RuntimeError(
            "yfinance returned empty data — check tickers, date range, and network."
        )

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        logger.warning("yfinance didn't return data for: %s", missing)

    # drop anything with sparse coverage — these cause problems downstream
    coverage = prices.notna().mean()
    low_coverage = coverage[coverage < min_coverage].index.tolist()
    if low_coverage:
        logger.warning(
            "dropping %d ticker(s) with < %.0f%% coverage: %s",
            len(low_coverage), min_coverage * 100, low_coverage,
        )
        prices = prices.drop(columns=low_coverage)

    if prices.empty:
        raise RuntimeError(
            f"all tickers dropped after coverage filter ({min_coverage:.0%})"
        )

    # ffill first handles mid-series gaps (holidays etc), bfill handles leading NaNs
    prices = prices.ffill().bfill()
    prices = prices.dropna(how="all")

    logger.info("final price matrix: %d tickers × %d days", prices.shape[1], prices.shape[0])

    prices.to_parquet(path)
    logger.info("cached to %s", path)

    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_{i,t} = ln(P_{i,t} / P_{i,t-1}).

    Using log returns rather than simple returns because they're time-additive
    and better-behaved in regression. First row is dropped.
    """
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    return np.log(prices / prices.shift(1)).dropna(how="all")


def load_single(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = ".cache",
) -> pd.Series:
    """Load a single ticker as a Series — mostly used for SPY and VIX."""
    df = load([ticker], start=start, end=end, cache_dir=cache_dir)
    if ticker not in df.columns:
        raise KeyError(f"'{ticker}' not in downloaded data. Available: {df.columns.tolist()}")
    return df[ticker].rename(ticker)
