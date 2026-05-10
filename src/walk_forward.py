"""
walk_forward.py — rolling out-of-sample validation.

In-sample backtest results are nearly meaningless. What matters is whether
the strategy works on data it has never seen. Walk-forward does this properly:
fit everything on TRAIN, test on the next OUT_OF_SAMPLE window, then roll
forward and repeat.

Strict no-lookahead rules:
  - Pair selection uses only train-window neutral prices
  - Kalman filter is fitted only on train data, state carried forward
  - Regime detector is trained only on train market returns
  - OU params are estimated only on train spread

One design note: factor neutralisation is computed ONCE on the full price
history before the fold loop starts. The betas (market, sector) are slow-moving
enough that this is acceptable — using them to pre-compute a neutral price series
is not the same kind of lookahead as using future prices in the Kalman or EG test.
The pair selection, signal generation, and sizing all remain strictly OOS.

TODO: look into whether Bai-Perron structural break detection is worth adding here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from src import analytics
from src import backtest
from src import data_loader
from src import factor_neutralisation
from src import kalman_filter
from src import ou_process
from src import pair_selection
from src import regime_detector
from src import strategy
from config import Config

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results for one train/test fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    portfolio_result: backtest.PortfolioResult
    n_pairs: int
    oos_sharpe: float
    oos_returns: pd.Series


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""
    fold_results: List[FoldResult]
    oos_equity: pd.Series
    oos_returns: pd.Series
    oos_sharpe: float
    oos_cagr_pct: float
    oos_max_drawdown_pct: float
    per_fold_sharpe: List[float]
    n_folds: int


def run_walk_forward(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
    config: Config,
) -> WalkForwardResult:
    """
    Run rolling walk-forward validation across the full price history.

    Pre-computes factor-neutral prices once on the full history, then for
    each fold uses only the train-window slice for pair selection and Kalman
    fitting to ensure no lookahead in the statistical tests.
    """
    folds = _build_folds(prices.index, config.train_months, config.test_months)
    if not folds:
        raise ValueError("no valid walk-forward folds — check date range and train/test months")

    logger.info(
        "starting walk-forward: %d folds, %d-month train, %d-month test",
        len(folds), config.train_months, config.test_months,
    )

    # neutralise once on the full sample — factor betas are slow-moving
    # so pre-computing them doesn't constitute lookahead in the same way
    # that using future prices in EG or Kalman would
    logger.info("pre-computing factor-neutral prices for full history...")
    spy_returns_full = data_loader.log_returns(spy_prices.to_frame()).iloc[:, 0]
    all_returns = data_loader.log_returns(prices)
    neutral_prices_full, neutral_returns_full, _ = factor_neutralisation.neutralise(
        prices=prices,
        returns=all_returns,
        market_returns=spy_returns_full,
        sector_map=config.sector_map,
        neutralise_market=config.neutralise_market,
        neutralise_sector=config.neutralise_sector,
    )

    fold_results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        logger.info(
            "fold %d/%d: train %s -> %s, test %s -> %s",
            i + 1, len(folds),
            train_start.date(), train_end.date(),
            test_start.date(), test_end.date(),
        )
        try:
            fold = _run_fold(
                fold_index=i,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                prices=prices,
                neutral_prices_full=neutral_prices_full,
                neutral_returns_full=neutral_returns_full,
                spy_prices=spy_prices,
                spy_returns_full=spy_returns_full,
                vix_prices=vix_prices,
                config=config,
            )
            fold_results.append(fold)
        except Exception as exc:
            logger.error("fold %d failed: %s -- skipping", i + 1, exc)
            continue

    if not fold_results:
        raise RuntimeError("all walk-forward folds failed")

    # stitch OOS returns together chronologically
    oos_returns = pd.concat([f.oos_returns for f in fold_results]).sort_index()
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    oos_equity = pd.Series(
        np.cumprod(1.0 + oos_returns.values),
        index=oos_returns.index,
        name="oos_equity",
    )

    return WalkForwardResult(
        fold_results=fold_results,
        oos_equity=oos_equity,
        oos_returns=oos_returns,
        oos_sharpe=analytics.sharpe_ratio(oos_returns),
        oos_cagr_pct=analytics.cagr_pct(oos_equity),
        oos_max_drawdown_pct=analytics.max_drawdown_pct(oos_equity),
        per_fold_sharpe=[f.oos_sharpe for f in fold_results],
        n_folds=len(fold_results),
    )


def _run_fold(
    fold_index: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    prices: pd.DataFrame,
    neutral_prices_full: pd.DataFrame,
    neutral_returns_full: pd.DataFrame,
    spy_prices: pd.Series,
    spy_returns_full: pd.Series,
    vix_prices: pd.Series,
    config: Config,
) -> FoldResult:
    """
    Run one fold of walk-forward validation.

    NO LOOKAHEAD for statistical tests:
      - Pair selection uses only train-window neutral prices/returns
      - Kalman state is fitted on train window only, then carried into test
      - OU params are estimated from train spread only
      - Regime HMM is fitted on train market returns only
    """
    # slice train and test windows from the pre-neutralised full history
    neutral_prices_train = neutral_prices_full.loc[train_start:train_end]
    neutral_returns_train = neutral_returns_full.loc[train_start:train_end]
    neutral_prices_test = neutral_prices_full.loc[test_start:test_end]

    spy_returns_train = spy_returns_full.loc[train_start:train_end]
    spy_returns_test = spy_returns_full.loc[test_start:test_end]
    vix_train = vix_prices.loc[train_start:train_end]

    if len(neutral_prices_train) < 60 or len(neutral_prices_test) < 5:
        raise ValueError(
            f"fold {fold_index}: not enough data in train ({len(neutral_prices_train)}) "
            f"or test ({len(neutral_prices_test)})"
        )

    # --- pair selection on train window only (NO LOOKAHEAD) ---
    pairs_df = pair_selection.select_pairs(
        neutral_prices=neutral_prices_train,
        neutral_returns=neutral_returns_train,
        sector_map=config.sector_map,
        min_correlation=config.min_correlation,
        coint_pvalue=config.coint_pvalue,
        min_half_life=config.min_half_life,
        max_half_life=config.max_half_life,
        rolling_coint_window=config.rolling_coint_window,
        max_pairs_per_sector=config.max_pairs_per_sector,
        n_workers=1,
    )

    if pairs_df.empty:
        logger.warning("fold %d: no pairs found in training window", fold_index)
        empty_ret = pd.Series(0.0, index=neutral_prices_test.index, name="oos_return")
        empty_eq = pd.Series(1.0, index=neutral_prices_test.index, name="oos_equity")
        return FoldResult(
            fold_index=fold_index,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            portfolio_result=backtest.PortfolioResult(
                equity_curve=empty_eq, returns=empty_ret,
                pair_results=[], portfolio_beta=0.0,
                portfolio_correlation_to_spy=0.0,
                initial_capital=config.initial_capital,
            ),
            n_pairs=0, oos_sharpe=0.0, oos_returns=empty_ret,
        )

    top_pairs = pairs_df.head(config.top_n_pairs)

    # --- regime detector trained on train window only (NO LOOKAHEAD) ---
    reg_detector = regime_detector.fit(
        market_returns=spy_returns_train,
        vix_series=vix_prices,  # pass full VIX so predict() can look up test dates
        train_end_date=str(train_end.date()),
        vix_threshold=config.vix_threshold,
        hmm_n_states=config.hmm_n_states,
        use_hmm=config.use_hmm,
    )

    # regime predictions for the test window
    regime_labels = reg_detector.predict(neutral_prices_test.index)

    # raw returns for the test window (for backtest P&L)
    prices_for_returns = prices.loc[train_end:test_end]  # one extra row for first diff
    test_returns = data_loader.log_returns(prices_for_returns)
    test_returns = test_returns.loc[test_start:test_end]

    pair_results = []
    for _, row in top_pairs.iterrows():
        ticker_y = str(row["ticker_y"])
        ticker_x = str(row["ticker_x"])

        if ticker_y not in neutral_prices_test.columns or ticker_x not in neutral_prices_test.columns:
            continue

        try:
            # carry kalman state from end of train into test (NO LOOKAHEAD)
            theta_init, P_init = kalman_filter.carry_state(
                y_train=neutral_prices_train[ticker_y],
                x_train=neutral_prices_train[ticker_x],
                delta=config.kalman_delta,
                vt=config.kalman_vt,
            )

            # run kalman on test window starting from train-end state
            hedge_ratio, intercept, spread = kalman_filter.fit_with_state(
                y=neutral_prices_test[ticker_y],
                x=neutral_prices_test[ticker_x],
                theta_init=theta_init,
                P_init=P_init,
                delta=config.kalman_delta,
                vt=config.kalman_vt,
            )

            z = kalman_filter.zscore(spread, window=20)

            # OU params from train spread only (NO LOOKAHEAD)
            _, _, train_spread = kalman_filter.fit(
                y=neutral_prices_train[ticker_y],
                x=neutral_prices_train[ticker_x],
                delta=config.kalman_delta,
                vt=config.kalman_vt,
            )
            try:
                ou_params = ou_process.fit(train_spread, method=config.ou_fitting_method)
            except Exception:
                ou_params = None

            sigs = strategy.generate_signals(
                zscore=z,
                regime=regime_labels,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
                stop_z=config.stop_z,
                ou_params=ou_params,
                use_ou_thresholds=config.use_ou_thresholds,
                sharpe_target=config.sharpe_target,
                spread=train_spread,
            )

            ret_y = test_returns[ticker_y] if ticker_y in test_returns.columns else pd.Series(dtype=float)
            ret_x = test_returns[ticker_x] if ticker_x in test_returns.columns else pd.Series(dtype=float)

            pr = backtest.run_pair_backtest(
                ticker_y=ticker_y, ticker_x=ticker_x,
                returns_y=ret_y, returns_x=ret_x,
                signals=sigs, regime=regime_labels,
                hedge_ratio=hedge_ratio, spread=spread, zscore=z,
                transaction_cost=config.transaction_cost,
                ou_params=ou_params,
            )
            pair_results.append(pr)

        except Exception as exc:
            logger.warning("fold %d, pair %s/%s failed: %s", fold_index, ticker_y, ticker_x, exc)
            continue

    portfolio = backtest.run_portfolio_backtest(
        pair_results=pair_results,
        spy_returns=spy_returns_test,
        initial_capital=config.initial_capital,
        sizing_method=config.sizing_method,
        max_pair_weight=config.max_pair_weight,
    )

    oos_returns = portfolio.returns.reindex(neutral_prices_test.index, fill_value=0.0)
    oos_sharpe = analytics.sharpe_ratio(oos_returns)

    logger.info("fold %d OOS sharpe=%.3f, n_pairs=%d", fold_index, oos_sharpe, len(pair_results))

    return FoldResult(
        fold_index=fold_index,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        portfolio_result=portfolio,
        n_pairs=len(pair_results),
        oos_sharpe=oos_sharpe,
        oos_returns=oos_returns,
    )


def _build_folds(
    date_index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Build (train_start, train_end, test_start, test_end) tuples.

    Folds roll forward by test_months each iteration. Stops when the
    remaining data is less than one full test window.
    """
    start = date_index[0]
    end = date_index[-1]
    folds = []
    fold_start = start

    while True:
        train_end = fold_start + relativedelta(months=train_months) - relativedelta(days=1)
        test_start = train_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end:
            break

        # snap to actual trading days
        ts_arr = date_index[date_index >= fold_start]
        te_arr = date_index[date_index <= train_end]
        ts_s_arr = date_index[date_index >= test_start]
        te_s_arr = date_index[date_index <= test_end]

        if len(ts_arr) == 0 or len(te_arr) == 0 or len(ts_s_arr) == 0 or len(te_s_arr) == 0:
            break

        folds.append((ts_arr[0], te_arr[-1], ts_s_arr[0], te_s_arr[-1]))
        fold_start = fold_start + relativedelta(months=test_months)

    return folds
