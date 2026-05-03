"""
regime_detector.py — block new pair entries during bad market regimes.

Pairs trading tends to fall apart when volatility spikes. Spreads that
were cointegrated for two years suddenly blow through stop-losses, because
the relationships that held in normal conditions don't hold during stress.
Better to sit on your hands and wait it out.

Two methods, combined with AND (both must agree before calling it "normal"):

VIX threshold: regime = 1 when VIX < 25, else 0. Simple and always available.

HMM on realised vol: fit a 2-state Gaussian HMM to rolling 20-day realised
volatility of SPY. The low-vol state = tradeable, high-vol state = pause.
The HMM is fitted on TRAIN data only — no lookahead.

When regime = 0, new entries are blocked but existing positions stay open.
Forcing exits during a risk-off period is often worse than just holding.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn not installed — falling back to VIX-only regime detection")


class RegimeDetector:
    """
    Two-method market regime classifier. VIX threshold AND HMM must both
    agree before a day is classified as tradeable.

    Fit on training data, predict on test data. The HMM is never
    re-trained on test data — that would introduce lookahead.
    """

    def __init__(
        self,
        vix_threshold: float = 25.0,
        hmm_n_states: int = 2,
        use_hmm: bool = True,
    ) -> None:
        self.vix_threshold = vix_threshold
        self.hmm_n_states = hmm_n_states
        self.use_hmm = use_hmm and _HMMLEARN_AVAILABLE

        self._hmm: Optional[GaussianHMM] = None
        self._hmm_low_vol_state: int = 0
        self._train_end: Optional[pd.Timestamp] = None
        self._full_vix: Optional[pd.Series] = None
        self._full_market_returns: Optional[pd.Series] = None

        if use_hmm and not _HMMLEARN_AVAILABLE:
            logger.warning("use_hmm=True but hmmlearn isn't installed — VIX only")

    def fit(
        self,
        market_returns: pd.Series,
        vix_series: pd.Series,
        train_end_date: Optional[str] = None,
    ) -> "RegimeDetector":
        """
        Fit the HMM on training data. Store the full VIX series for predict().

        NO LOOKAHEAD: HMM only sees market_returns up to train_end_date.
        """
        self._full_vix = vix_series.copy()
        # store the full market returns so _predict_hmm can compute rolling vol
        # on the test window using the same feature as training
        self._full_market_returns = market_returns.copy()

        if train_end_date is not None:
            train_end = pd.Timestamp(train_end_date)
            # slice to training window only
            train_returns = market_returns[market_returns.index <= train_end]
        else:
            train_returns = market_returns.copy()
            train_end = market_returns.index[-1]

        self._train_end = train_end

        if self.use_hmm:
            self._fit_hmm(train_returns)

        logger.info(
            "regime detector fitted on %d training days (end=%s), use_hmm=%s",
            len(train_returns), train_end.date(), self.use_hmm,
        )
        return self

    def predict(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Return regime labels {0=risk-off, 1=normal} for the given dates.

        Combined regime = VIX_regime AND HMM_regime. Conservative — both
        methods need to agree that conditions are normal before we trade.
        """
        if self._full_vix is None:
            raise RuntimeError("call fit() before predict()")

        vix_on_dates = self._full_vix.reindex(dates, method="ffill")
        vix_regime = (vix_on_dates < self.vix_threshold).astype(int)

        if not self.use_hmm or self._hmm is None:
            return vix_regime.rename("regime")

        hmm_regime = self._predict_hmm(dates)
        regime = (vix_regime & hmm_regime).astype(int).rename("regime")

        risk_off_pct = (1 - regime).mean() * 100
        logger.debug("regime prediction: %.1f%% risk-off over %d dates", risk_off_pct, len(dates))

        return regime

    def _fit_hmm(self, market_returns: pd.Series) -> None:
        """
        Fit a 2-state Gaussian HMM on rolling 20-day realised vol of SPY.

        Using realised vol as the feature rather than raw returns because
        it's more stationary and captures the clustering of volatility better.

        NO LOOKAHEAD: only sees market_returns from the training window.
        """
        rv = (
            market_returns
            .rolling(window=20, min_periods=10)
            .std(ddof=1)
            .dropna()
            * np.sqrt(252)
        )

        if len(rv) < 60:
            logger.warning("only %d obs for HMM fitting — too few, falling back to VIX only", len(rv))
            self.use_hmm = False
            return

        X = rv.values.reshape(-1, 1)

        try:
            hmm = GaussianHMM(
                n_components=self.hmm_n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=1e-4,
            )
            hmm.fit(X)

            if not hmm.monitor_.converged:
                logger.warning("HMM didn't converge — falling back to VIX only")
                self.use_hmm = False
                return

            self._hmm = hmm
            self._rv_train = rv.copy()

            # figure out which state is "low vol" by comparing state means
            state_means = hmm.means_.flatten()
            self._hmm_low_vol_state = int(np.argmin(state_means))
            logger.info(
                "HMM fitted: state vol means = [%.3f, %.3f], low-vol state = %d",
                state_means[0], state_means[1], self._hmm_low_vol_state,
            )

        except Exception as exc:
            logger.warning("HMM fitting failed: %s — falling back to VIX only", exc)
            self.use_hmm = False

    def _predict_hmm(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Decode HMM states using the same rolling-realized-vol feature as training.

        NO LOOKAHEAD: HMM parameters (_hmm) come from training data only.
        We compute rolling vol from the stored full market returns — the HMM
        only sees vol values, not future returns.

        Using VIX directly here would be wrong because the HMM was trained on
        annualised realized vol (~0.1-0.5 range), not VIX (~10-80). The scale
        mismatch would make the HMM classify everything as high-vol.
        """
        if self._hmm is None:
            return pd.Series(1, index=dates, name="hmm_regime")

        if not hasattr(self, "_full_market_returns") or self._full_market_returns is None:
            return pd.Series(1, index=dates, name="hmm_regime")

        try:
            # recompute rolling vol on the full returns series (train + test)
            full_rv = (
                self._full_market_returns
                .rolling(window=20, min_periods=10)
                .std(ddof=1)
                .dropna()
                * np.sqrt(252)
            )
            rv_on_dates = full_rv.reindex(dates, method="ffill").dropna()
            if rv_on_dates.empty:
                return pd.Series(1, index=dates, name="hmm_regime")

            states = self._hmm.predict(rv_on_dates.values.reshape(-1, 1))
        except Exception as exc:
            logger.warning("HMM decode failed: %s -- defaulting to regime=1", exc)
            return pd.Series(1, index=dates, name="hmm_regime")

        is_low_vol = (states == self._hmm_low_vol_state).astype(int)
        hmm_regime = pd.Series(is_low_vol, index=rv_on_dates.index, name="hmm_regime")
        return hmm_regime.reindex(dates, fill_value=0)


def fit(
    market_returns: pd.Series,
    vix_series: pd.Series,
    train_end_date: Optional[str] = None,
    vix_threshold: float = 25.0,
    hmm_n_states: int = 2,
    use_hmm: bool = True,
) -> RegimeDetector:
    """Convenience constructor — build and fit a RegimeDetector in one call."""
    detector = RegimeDetector(
        vix_threshold=vix_threshold,
        hmm_n_states=hmm_n_states,
        use_hmm=use_hmm,
    )
    return detector.fit(market_returns, vix_series, train_end_date=train_end_date)
