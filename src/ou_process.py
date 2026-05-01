"""
ou_process.py — Ornstein-Uhlenbeck parameter fitting and optimal threshold derivation.

The spread is modelled as a continuous-time OU process:
    dS_t = κ(μ − S_t)dt + σ dW_t

κ controls how fast it mean-reverts, μ is the long-run mean, σ is the noise,
and σ_eq = σ/√(2κ) is the equilibrium standard deviation.

Two fitting methods:
  OLS  — discretise as AR(1) and recover parameters from the coefficients. Fast.
  MLE  — maximise the exact likelihood of the discretised process. More accurate.

Optimal thresholds come from Avellaneda & Lee (2010) — the entry threshold that
maximises expected Sharpe for a round-trip trade under OU dynamics is:
    s_entry = −σ_eq · √(sharpe_target²/(2κ) + 1)
    s_exit  = 0  (exit when spread returns to mean)

These are in spread units, not z-score units. Divide by rolling spread std to convert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    """
    Fitted parameters of an Ornstein-Uhlenbeck process.

    kappa:     mean-reversion speed (per trading day)
    mu:        long-run mean
    sigma:     diffusion coefficient
    sigma_eq:  equilibrium std = σ/√(2κ)
    half_life: ln(2)/κ in trading days
    fitting_method: "ols" or "mle"
    log_likelihood: MLE objective value (NaN for OLS)
    """
    kappa: float
    mu: float
    sigma: float
    sigma_eq: float
    half_life: float
    fitting_method: str
    log_likelihood: float = float("nan")

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa:.6f}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma:.6f}")

    def __repr__(self) -> str:
        return (
            f"OUParams(κ={self.kappa:.4f}, μ={self.mu:.4f}, "
            f"σ={self.sigma:.4f}, σ_eq={self.sigma_eq:.4f}, "
            f"HL={self.half_life:.1f}d, method={self.fitting_method})"
        )


def fit(
    spread: pd.Series,
    method: str = "mle",
    dt: float = 1.0 / 252.0,
) -> OUParams:
    """
    Fit OU parameters to a spread series using OLS or MLE.

    Both methods fit the AR(1) discretisation:
        S_t = a + b·S_{t-1} + ε_t
    where a = μ(1 − e^{−κΔt}), b = e^{−κΔt}.
    OLS solves it directly; MLE maximises the proper likelihood.
    MLE falls back to OLS if the optimiser doesn't converge.
    """
    if method not in ("ols", "mle"):
        raise ValueError(f"method must be 'ols' or 'mle', got '{method}'")

    s = spread.dropna().values.astype(float)
    if len(s) < 30:
        raise ValueError(f"spread has only {len(s)} observations — need at least 30")

    return _fit_ols(s, dt) if method == "ols" else _fit_mle(s, dt)


def optimal_thresholds(
    ou_params: OUParams,
    sharpe_target: float = 1.0,
) -> Tuple[float, float]:
    """
    Derive entry and exit thresholds in spread units from the OU parameters.

    Per the Avellaneda & Lee (2010) optimal stopping solution, the entry
    threshold that maximises expected Sharpe on a round-trip trade is:

        s_entry = −σ_eq · √(sharpe_target²/(2κ) + 1)
        s_exit  = 0  (exit at long-run mean)

    Returns (entry_threshold, exit_threshold) as floats. entry is negative
    for the long side — symmetric short entry is at +|entry|.
    """
    entry = -ou_params.sigma_eq * np.sqrt(sharpe_target ** 2 / (2.0 * ou_params.kappa) + 1.0)
    return float(entry), 0.0


def plot_diagnostics(spread: pd.Series, ou_params: OUParams) -> plt.Figure:
    """Three-panel diagnostic: spread with bands, ACF, and QQ-plot of residuals."""
    from statsmodels.graphics.tsaplots import plot_acf

    s = spread.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"OU fit  |  κ={ou_params.kappa:.3f}  μ={ou_params.mu:.4f}  "
        f"σ_eq={ou_params.sigma_eq:.4f}  HL={ou_params.half_life:.1f}d",
        fontsize=11,
    )

    # spread with ±σ_eq and ±2σ_eq bands
    ax = axes[0]
    ax.plot(s.index, s.values, lw=0.8, color="steelblue", label="spread")
    mu, sig = ou_params.mu, ou_params.sigma_eq
    ax.axhline(mu, color="black", lw=1.0, ls="--", label="μ")
    ax.axhline(mu + sig, color="orange", lw=0.8, ls=":", label="±σ_eq")
    ax.axhline(mu - sig, color="orange", lw=0.8, ls=":")
    ax.axhline(mu + 2 * sig, color="red", lw=0.8, ls=":", label="±2σ_eq")
    ax.axhline(mu - 2 * sig, color="red", lw=0.8, ls=":")
    ax.set_title("spread + OU bands")
    ax.legend(fontsize=7)

    # ACF — should decay geometrically if the OU fit is reasonable
    plot_acf(s.values, lags=40, ax=axes[1], zero=False, alpha=0.05)
    axes[1].set_title("ACF of spread")

    # QQ-plot of standardised residuals
    ax = axes[2]
    residuals = (s.values - ou_params.mu) / ou_params.sigma_eq
    sorted_resid = np.sort(residuals)
    theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
    ax.scatter(theoretical, sorted_resid, s=4, alpha=0.5, color="steelblue")
    mn = min(theoretical.min(), sorted_resid.min())
    mx = max(theoretical.max(), sorted_resid.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    ax.set_title("QQ-plot (standardised residuals)")
    ax.set_xlabel("theoretical")
    ax.set_ylabel("sample")

    plt.tight_layout()
    return fig


def _fit_ols(s: np.ndarray, dt: float) -> OUParams:
    """
    OLS on the AR(1) discretisation: S_t = a + b·S_{t-1} + ε.

    Recovery:
        κ = −ln(b) / Δt
        μ = a / (1 − b)
        σ = std(ε) · √(2κ / (1 − b²))
    """
    s_lag = s[:-1]
    s_curr = s[1:]

    A = np.column_stack([np.ones_like(s_lag), s_lag])
    a, b = np.linalg.lstsq(A, s_curr, rcond=None)[0]

    if b <= 0 or b >= 1:
        logger.warning("AR(1) coefficient b=%.4f outside (0,1) — clamping", b)
        b = float(np.clip(b, 1e-3, 1.0 - 1e-3))

    kappa = -np.log(b) / dt
    mu = a / (1.0 - b)
    residuals = s_curr - (a + b * s_lag)
    sigma = float(np.std(residuals, ddof=1) * np.sqrt(2.0 * kappa / (1.0 - b ** 2)))
    sigma = max(sigma, 1e-9)
    sigma_eq = sigma / np.sqrt(2.0 * kappa)

    return OUParams(
        kappa=float(kappa),
        mu=float(mu),
        sigma=float(sigma),
        sigma_eq=float(sigma_eq),
        half_life=float(np.log(2.0) / kappa / dt),  # convert from years to days
        fitting_method="ols",
    )


def _fit_mle(s: np.ndarray, dt: float) -> OUParams:
    """
    MLE for the discretised OU process.

    Log-likelihood:
        L = −(n/2)ln(2π) − (n/2)ln(Var_ε)
            − Σ[(S_t − μ_c − e^{−κΔt}(S_{t-1} − μ_c))² / (2·Var_ε)]

    where Var_ε = σ²(1 − e^{−2κΔt}) / (2κ).

    Optimised with L-BFGS-B. Falls back to OLS if convergence fails.
    """
    s_lag = s[:-1]
    s_curr = s[1:]
    n = len(s_curr)

    def neg_log_likelihood(params: np.ndarray) -> float:
        kappa_dt, mu_c, sigma = params
        if kappa_dt <= 0 or sigma <= 0:
            return 1e12
        exp_kdt = np.exp(-kappa_dt)
        var_eps = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa_dt)) / (2.0 * kappa_dt)
        if var_eps <= 0:
            return 1e12
        residuals = s_curr - mu_c - exp_kdt * (s_lag - mu_c)
        return (
            0.5 * n * np.log(2.0 * np.pi)
            + 0.5 * n * np.log(var_eps)
            + 0.5 * np.sum(residuals ** 2) / var_eps
        )

    # warm-start from OLS so the optimiser doesn't start blind
    try:
        ols = _fit_ols(s, dt)
        x0 = np.array([ols.kappa * dt, ols.mu, ols.sigma])
    except Exception:
        x0 = np.array([1.0 * dt, float(np.mean(s)), float(np.std(s, ddof=1))])

    try:
        opt = minimize(
            neg_log_likelihood,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(1e-6, None), (None, None), (1e-9, None)],
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not opt.success:
            logger.warning("MLE didn't converge (status %d: %s) — falling back to OLS", opt.status, opt.message)
            return _fit_ols(s, dt)

        # x0[0] is kappa * dt, so kappa per day = x0[0] / dt
        kappa_dt_mle, mu_mle, sigma_mle = opt.x
        kappa_per_day = kappa_dt_mle / dt
        sigma_eq = sigma_mle / np.sqrt(2.0 * kappa_per_day)
        half_life = np.log(2.0) / kappa_per_day

        return OUParams(
            kappa=float(kappa_per_day),
            mu=float(mu_mle),
            sigma=float(sigma_mle),
            sigma_eq=float(sigma_eq),
            half_life=float(half_life),
            fitting_method="mle",
            log_likelihood=float(-opt.fun),
        )

    except Exception as exc:
        logger.warning("MLE raised an exception: %s — falling back to OLS", exc)
        return _fit_ols(s, dt)
