"""
kalman_filter.py — online Kalman filter for a time-varying hedge ratio.

The problem with a static OLS hedge ratio is that equity relationships drift.
By day 252, the β you estimated at t=0 is almost certainly wrong. The Kalman
filter treats [β_t, μ_t] as a latent state that evolves as a random walk and
updates it with each new observation. It adapts to structural drift without
needing to be periodically re-estimated.

State-space model:
    observation:  y_t = [x_t, 1] · θ_t + ε_t,    ε_t ~ N(0, R)
    state:        θ_t = θ_{t-1} + w_t,            w_t ~ N(0, Q)

    θ_t = [β_t, μ_t]ᵀ
    Q   = (δ/(1−δ)) · I₂  — process noise, controls how fast β can drift
    R   = v_t              — observation noise

Predict / update cycle (derivation in the README):
    P_{t|t-1} = P_{t-1} + Q
    ν_t       = y_t − [x_t, 1] · θ_{t-1}           (innovation)
    S_t       = F_t · P_{t|t-1} · F_tᵀ + R          (innovation covariance)
    K_t       = P_{t|t-1} · F_tᵀ / S_t              (Kalman gain)
    θ_t       = θ_{t-1} + K_t · ν_t
    P_t       = (I − K_t · F_t) · P_{t|t-1}

The inner loop is pure NumPy — no pandas row iteration. That matters when
you're running this across 10 pairs × 1500 trading days.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    vt: float = 1e-3,
    warmup_periods: int = 0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Run the Kalman filter and return (hedge_ratio, intercept, spread).

    Warm-starts θ with OLS on the first min(30, n//4) observations so the
    filter doesn't start with a meaningless hedge ratio and take 50 days to
    recover — the kalman warmup period is bad enough as it is.

    warmup_periods: if > 0, trims the first N rows from the output
    (useful when carrying state across walk-forward folds).
    """
    common = y.index.intersection(x.index)
    y_vals = y.loc[common].values.astype(float)
    x_vals = x.loc[common].values.astype(float)
    n = len(y_vals)

    if n < 4:
        raise ValueError(f"only {n} overlapping observations — need at least 4")

    # Q controls how much the hedge ratio is allowed to move each day
    # small delta = slow adaptation, large delta = fast but noisy
    Q = (delta / (1.0 - delta)) * np.eye(2)
    R_obs = float(vt)
    I2 = np.eye(2)

    # warm-start with OLS so we're not starting from zero
    warmup_n = min(30, max(4, n // 4))
    X_init = np.column_stack([x_vals[:warmup_n], np.ones(warmup_n)])
    try:
        theta, _, _, _ = np.linalg.lstsq(X_init, y_vals[:warmup_n], rcond=None)
    except np.linalg.LinAlgError:
        theta = np.array([1.0, 0.0])
        logger.warning("OLS warmup failed — initialising θ = [1, 0]")

    P = np.eye(2)

    beta_arr = np.empty(n)
    mu_arr = np.empty(n)

    # main loop — pure numpy, no pandas overhead
    # NO LOOKAHEAD: at step t, θ is updated using only x_t, y_t and the prior
    # state from t-1. No future observations enter the filter.
    for t in range(n):
        P_pred = P + Q

        F = np.array([x_vals[t], 1.0])
        innovation = y_vals[t] - F @ theta
        S = float(F @ P_pred @ F) + R_obs
        K = (P_pred @ F) / S

        theta = theta + K * innovation
        P = (I2 - np.outer(K, F)) @ P_pred

        beta_arr[t] = theta[0]
        mu_arr[t] = theta[1]

    hedge_ratio = pd.Series(beta_arr, index=common, name="hedge_ratio")
    intercept = pd.Series(mu_arr, index=common, name="intercept")
    # spread = static OLS version: y - hedge_ratio * x
    # kalman version subtracts the intercept too — kalman is better
    spread = pd.Series(y_vals - beta_arr * x_vals - mu_arr, index=common, name="kalman_spread")

    if warmup_periods > 0 and warmup_periods < n:
        idx = common[warmup_periods:]
        hedge_ratio = hedge_ratio.loc[idx]
        intercept = intercept.loc[idx]
        spread = spread.loc[idx]

    return hedge_ratio, intercept, spread


def zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling z-score of the spread: z_t = (S_t − rolling_mean) / rolling_std.

    min_periods=window means we don't get artificially inflated z-scores early
    on from partial windows — the first window-1 values are just NaN.

    NO LOOKAHEAD: rolling() only looks backwards.
    """
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std = spread.rolling(window=window, min_periods=window).std(ddof=1)
    z = (spread - roll_mean) / roll_std.replace(0.0, np.nan)
    z.name = "zscore"
    return z


def carry_state(
    y_train: pd.Series,
    x_train: pd.Series,
    delta: float = 1e-4,
    vt: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the filter on training data and return the final (θ, P) state.

    Used in walk-forward validation so the test window picks up exactly
    where training left off — no cold start, no re-fitting on test data.

    NO LOOKAHEAD: only trains on y_train, x_train. The returned state
    is then used to initialise fit_with_state() on the test window.
    """
    common = y_train.index.intersection(x_train.index)
    y_vals = y_train.loc[common].values.astype(float)
    x_vals = x_train.loc[common].values.astype(float)
    n = len(y_vals)

    Q = (delta / (1.0 - delta)) * np.eye(2)
    R_obs = float(vt)
    I2 = np.eye(2)

    warmup_n = min(30, max(4, n // 4))
    X_init = np.column_stack([x_vals[:warmup_n], np.ones(warmup_n)])
    try:
        theta, _, _, _ = np.linalg.lstsq(X_init, y_vals[:warmup_n], rcond=None)
    except np.linalg.LinAlgError:
        theta = np.array([1.0, 0.0])

    P = np.eye(2)

    for t in range(n):
        P_pred = P + Q
        F = np.array([x_vals[t], 1.0])
        K = (P_pred @ F) / (float(F @ P_pred @ F) + R_obs)
        theta = theta + K * (y_vals[t] - F @ theta)
        P = (I2 - np.outer(K, F)) @ P_pred

    return theta.copy(), P.copy()


def fit_with_state(
    y: pd.Series,
    x: pd.Series,
    theta_init: np.ndarray,
    P_init: np.ndarray,
    delta: float = 1e-4,
    vt: float = 1e-3,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Run the filter on test data starting from a pre-existing (θ, P) state.

    NO LOOKAHEAD: theta_init and P_init come from carry_state() which only
    saw training data. The filter then updates forward using test observations
    one at a time.
    """
    common = y.index.intersection(x.index)
    y_vals = y.loc[common].values.astype(float)
    x_vals = x.loc[common].values.astype(float)
    n = len(y_vals)

    Q = (delta / (1.0 - delta)) * np.eye(2)
    R_obs = float(vt)
    I2 = np.eye(2)

    theta = theta_init.copy()
    P = P_init.copy()

    beta_arr = np.empty(n)
    mu_arr = np.empty(n)

    for t in range(n):
        P_pred = P + Q
        F = np.array([x_vals[t], 1.0])
        K = (P_pred @ F) / (float(F @ P_pred @ F) + R_obs)
        theta = theta + K * (y_vals[t] - F @ theta)
        P = (I2 - np.outer(K, F)) @ P_pred
        beta_arr[t] = theta[0]
        mu_arr[t] = theta[1]

    hedge_ratio = pd.Series(beta_arr, index=common, name="hedge_ratio")
    intercept = pd.Series(mu_arr, index=common, name="intercept")
    spread = pd.Series(y_vals - beta_arr * x_vals - mu_arr, index=common, name="kalman_spread")
    return hedge_ratio, intercept, spread
