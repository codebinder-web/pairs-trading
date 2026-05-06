"""
strategy.py — stateful signal generator for a single pair.

Signal convention:
    +1  long Y, short X   (spread below lower entry band)
    -1  short Y, long X   (spread above upper entry band)
     0  flat

Signals are computed at close of day t and executed at open of t+1.
The 1-day lag is enforced in backtest.py via signal.shift(1), not here.

Two threshold modes:
    Fixed z-score thresholds (entry_z, exit_z, stop_z) from config.
    OU-optimal thresholds from ou_process.optimal_thresholds — derived from
    the Avellaneda & Lee stopping problem, converted to z-score units.

The state machine is an explicit Python loop rather than vectorised apply
because the current position feeds into the next decision. Tried to vectorise
it at some point but the flip logic got messy — the loop is easier to follow.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.ou_process import OUParams, optimal_thresholds  # noqa: E402 — absolute import from project root

logger = logging.getLogger(__name__)


def generate_signals(
    zscore: pd.Series,
    regime: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    ou_params: Optional[OUParams] = None,
    use_ou_thresholds: bool = False,
    sharpe_target: float = 1.0,
    spread: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Run the signal state machine over a z-score series.

    State transitions:
      FLAT (0):
        z < −entry_z and regime=1  →  enter LONG  (+1)
        z > +entry_z and regime=1  →  enter SHORT (-1)
      LONG (+1):
        |z| > stop_z               →  stop-loss → FLAT
        z > +entry_z               →  flip to SHORT
        |z| < exit_z               →  take profit → FLAT
      SHORT (-1):
        |z| > stop_z               →  stop-loss → FLAT
        z < −entry_z               →  flip to LONG
        |z| < exit_z               →  take profit → FLAT

    New entries are blocked when regime=0. Existing positions stay open —
    forced exits during a crisis tend to be worse than holding.
    """
    common = zscore.index.intersection(regime.index)
    if len(common) == 0:
        logger.warning("zscore and regime have no overlapping dates — returning all-flat signals")
        return pd.Series(0, index=zscore.index, dtype=int, name="signal")

    z = zscore.loc[common].copy()
    reg = regime.loc[common].copy()

    # pass spread as-is — _resolve_thresholds only needs its std, not alignment with common
    effective_entry_z, effective_exit_z = _resolve_thresholds(
        z=z,
        spread=spread,
        ou_params=ou_params,
        use_ou_thresholds=use_ou_thresholds,
        sharpe_target=sharpe_target,
        entry_z=entry_z,
        exit_z=exit_z,
    )

    logger.debug(
        "generating signals: entry_z=%.2f, exit_z=%.2f, stop_z=%.2f, use_ou=%s, n=%d",
        effective_entry_z, effective_exit_z, stop_z, use_ou_thresholds, len(z),
    )

    z_arr = z.values.astype(float)
    reg_arr = reg.values.astype(int)
    n = len(z_arr)
    signals = np.zeros(n, dtype=int)
    current = 0  # current position

    for t in range(n):
        zt = z_arr[t]
        if np.isnan(zt):
            signals[t] = current  # hold through NaN — could be a market holiday
            continue

        abs_z = abs(zt)

        if current == 0:
            if reg_arr[t] == 1:
                if zt < -effective_entry_z:
                    current = 1
                elif zt > effective_entry_z:
                    current = -1

        else:
            if abs_z > stop_z:
                current = 0  # stop out

            elif current == 1:  # long
                if zt > effective_entry_z:
                    current = -1  # flip to short
                elif abs_z < effective_exit_z:
                    current = 0  # take profit

            else:  # short
                if zt < -effective_entry_z:
                    current = 1  # flip to long
                elif abs_z < effective_exit_z:
                    current = 0  # take profit

        signals[t] = current

    result = pd.Series(signals, index=common, name="signal")

    n_long = (result == 1).sum()
    n_short = (result == -1).sum()
    logger.debug(
        "signals done: %d long days, %d short days, %d flat days",
        n_long, (result == -1).sum(), (result == 0).sum(),
    )

    return result.reindex(zscore.index, fill_value=0)


def _resolve_thresholds(
    z: pd.Series,
    spread: Optional[pd.Series],
    ou_params: Optional[OUParams],
    use_ou_thresholds: bool,
    sharpe_target: float,
    entry_z: float,
    exit_z: float,
) -> Tuple[float, float]:
    """
    Return the effective (entry_z, exit_z) thresholds to use.

    If use_ou_thresholds=True and ou_params is available, converts the
    OU spread-unit thresholds into z-score units by dividing by spread std.
    Falls back to config thresholds if anything goes wrong.
    """
    if not use_ou_thresholds or ou_params is None:
        return entry_z, exit_z

    try:
        # per the avellaneda & lee paper this threshold maximises expected return
        ou_entry_spread, ou_exit_spread = optimal_thresholds(ou_params, sharpe_target=sharpe_target)

        spread_std = float(spread.std(ddof=1)) if spread is not None and spread.std(ddof=1) > 1e-9 else ou_params.sigma_eq
        if spread_std <= 0:
            return entry_z, exit_z

        entry_z_ou = abs(ou_entry_spread) / spread_std
        exit_z_ou = abs(ou_exit_spread) / spread_std if abs(ou_exit_spread) > 1e-9 else 0.0

        # sanity check — if the OU-derived threshold is nonsensical, fall back
        if not (0.5 <= entry_z_ou <= 5.0):
            logger.debug("OU entry_z=%.3f outside [0.5, 5.0] — using config value %.2f", entry_z_ou, entry_z)
            return entry_z, exit_z

        return float(entry_z_ou), float(exit_z_ou)

    except Exception as exc:
        logger.warning("OU threshold derivation failed: %s — using config thresholds", exc)
        return entry_z, exit_z
