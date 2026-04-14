"""
config.py — everything lives here, nothing hardcoded anywhere else.

If you want to tweak a parameter, change it here. Downstream modules
receive a Config instance and just read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# 40 tickers across 8 sectors — intentionally diversified so we get
# intra-sector pairs (energy vs energy) but not only those
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    "XOM": "Energy",    "CVX": "Energy",    "COP": "Energy",
    "SLB": "Energy",    "EOG": "Energy",
    "JPM": "Financials","BAC": "Financials","GS": "Financials",
    "MS":  "Financials","C":   "Financials",
    "MSFT":"Technology","GOOGL":"Technology","META":"Technology",
    "AAPL":"Technology","AMZN":"Technology",
    "KO":  "Consumer Staples","PEP":"Consumer Staples",
    "PG":  "Consumer Staples","CL": "Consumer Staples",
    "KMB": "Consumer Staples",
    "BA":  "Industrials","LMT":"Industrials","RTX":"Industrials",
    "NOC": "Industrials","GD": "Industrials",
    "JNJ": "Healthcare","PFE":"Healthcare","MRK":"Healthcare",
    "ABT": "Healthcare","BMY":"Healthcare",
    "NEE": "Utilities", "DUK":"Utilities","SO": "Utilities",
    "AEP": "Utilities", "EXC":"Utilities",
    "LIN": "Materials", "APD":"Materials","ECL":"Materials",
    "DD":  "Materials", "NEM":"Materials",
}

DEFAULT_TICKERS: List[str] = list(DEFAULT_SECTOR_MAP.keys())


@dataclass
class Config:
    """
    Single source of truth for every parameter in the system.

    Deliberately keeping this flat — one class, all values visible at a glance.
    The validation in __post_init__ catches the obvious mistakes early rather
    than letting nonsense values propagate through a six-hour backtest.
    """

    # universe
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    market_ticker: str = "SPY"

    # factor neutralisation — strip out market and sector beta before
    # testing cointegration, otherwise we're just finding stocks that
    # both load heavily on SPY and calling it a pair
    neutralise_market: bool = True
    neutralise_sector: bool = True
    sector_map: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_SECTOR_MAP)
    )

    # pair selection — these are looser than you might expect because
    # factor neutralisation already does a lot of the heavy lifting
    min_correlation: float = 0.25
    coint_pvalue: float = 0.10
    min_half_life: int = 3
    max_half_life: int = 90
    rolling_coint_window: int = 126  # ~6 months for structural break check

    # OU process fitting
    ou_fitting_method: str = "mle"   # "mle" or "ols" — mle is more accurate
    sharpe_target: float = 1.0       # used to derive optimal entry threshold

    # kalman filter — delta controls how fast the hedge ratio can drift,
    # vt is observation noise. these are pretty standard starting values
    kalman_delta: float = 1e-4
    kalman_vt: float = 1e-3

    # regime detection
    vix_threshold: float = 25.0
    hmm_n_states: int = 2
    use_hmm: bool = True

    # signal thresholds — if use_ou_thresholds is True these are fallbacks only
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    use_ou_thresholds: bool = True

    # position sizing — half kelly by default, full kelly is too aggressive
    # TODO: test whether half-kelly or full-kelly works better here
    sizing_method: str = "half_kelly"   # "equal", "kelly", "half_kelly"
    max_pair_weight: float = 0.15

    # backtest
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.0010   # 10 bps one-way
    max_pairs_per_sector: int = 2

    # walk-forward windows
    train_months: int = 12
    test_months: int = 3
    top_n_pairs: int = 10

    # sensitivity grid — entry_z vs max_half_life
    sensitivity_entry_z: List[float] = field(
        default_factory=lambda: [1.5, 2.0, 2.5, 3.0]
    )
    sensitivity_max_hl: List[int] = field(
        default_factory=lambda: [30, 45, 60, 90]
    )

    results_dir: str = "results"

    def __post_init__(self) -> None:
        if self.exit_z >= self.entry_z:
            raise ValueError(
                f"exit_z ({self.exit_z}) must be less than entry_z ({self.entry_z})"
            )
        if self.stop_z <= self.entry_z:
            raise ValueError(
                f"stop_z ({self.stop_z}) must be greater than entry_z ({self.entry_z})"
            )
        if self.ou_fitting_method not in ("mle", "ols"):
            raise ValueError(
                f"ou_fitting_method must be 'mle' or 'ols', got '{self.ou_fitting_method}'"
            )
        if self.sizing_method not in ("equal", "kelly", "half_kelly"):
            raise ValueError(
                f"sizing_method must be 'equal', 'kelly', or 'half_kelly', "
                f"got '{self.sizing_method}'"
            )
        if not 0.0 < self.max_pair_weight <= 1.0:
            raise ValueError(
                f"max_pair_weight must be in (0, 1], got {self.max_pair_weight}"
            )
        if self.train_months <= 0 or self.test_months <= 0:
            raise ValueError("train_months and test_months must be positive integers")
