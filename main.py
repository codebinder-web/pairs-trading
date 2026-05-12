"""
main.py — CLI entry point.

    python main.py                          # full backtest, print tearsheet
    python main.py --walk-forward           # add OOS validation
    python main.py --sensitivity --plot     # sensitivity heatmap, save charts
    python main.py --top-n 5 --plot         # run with 5 pairs, save HTML charts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# configure logging before anything else starts importing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pairs_trading.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import Config
from src import (
    analytics,
    backtest,
    data_loader,
    factor_neutralisation,
    kalman_filter,
    ou_process,
    pair_selection,
    regime_detector,
    strategy,
    visualisation,
    walk_forward,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairs Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--walk-forward", action="store_true",
                        help="run OOS walk-forward validation after the main backtest")
    parser.add_argument("--sensitivity", action="store_true",
                        help="run 2D sensitivity sweep over entry_z × max_half_life")
    parser.add_argument("--top-n", type=int, default=None,
                        help="number of pairs to trade (overrides config)")
    parser.add_argument("--plot", action="store_true",
                        help="save charts to results/ as HTML files")
    return parser.parse_args()


def run_pipeline(config: Config) -> backtest.PortfolioResult:
    """
    Full data → backtest pipeline.

    1. download prices (cached)
    2. factor neutralisation
    3. pair selection (parallelised)
    4. per-pair: kalman filter, OU fit, signal generation
    5. portfolio construction and backtest
    """
    os.makedirs(config.results_dir, exist_ok=True)
    logger.info("starting pipeline")
    t0 = time.perf_counter()

    # --- data loading ---
    all_tickers = config.tickers + [config.market_ticker]
    prices = data_loader.load(
        tickers=all_tickers,
        start=config.start_date,
        end=config.end_date,
        cache_dir=".cache",
    )

    market_prices = prices[config.market_ticker].dropna()
    stock_prices = prices.drop(columns=[config.market_ticker], errors="ignore")
    all_returns = data_loader.log_returns(prices)
    market_returns = all_returns[config.market_ticker].dropna()
    stock_returns = all_returns.drop(columns=[config.market_ticker], errors="ignore")

    # VIX for regime detection — falls back to a flat series if unavailable
    try:
        vix_df = data_loader.load(
            tickers=["^VIX"], start=config.start_date, end=config.end_date, cache_dir=".cache"
        )
        vix_series = vix_df["^VIX"].dropna()
    except Exception as vix_exc:
        logger.warning("couldn't load VIX: %s — defaulting to 20.0", vix_exc)
        vix_series = pd.Series(20.0, index=market_returns.index, name="^VIX")

    # --- factor neutralisation ---
    neutral_prices, neutral_returns, betas_df = factor_neutralisation.neutralise(
        prices=stock_prices,
        returns=stock_returns,
        market_returns=market_returns,
        sector_map=config.sector_map,
        neutralise_market=config.neutralise_market,
        neutralise_sector=config.neutralise_sector,
    )
    betas_df.to_csv(os.path.join(config.results_dir, "factor_betas.csv"))

    # --- pair selection ---
    selected_pairs = pair_selection.select_pairs(
        neutral_prices=neutral_prices,
        neutral_returns=neutral_returns,
        sector_map=config.sector_map,
        min_correlation=config.min_correlation,
        coint_pvalue=config.coint_pvalue,
        min_half_life=config.min_half_life,
        max_half_life=config.max_half_life,
        rolling_coint_window=config.rolling_coint_window,
        max_pairs_per_sector=config.max_pairs_per_sector,
    )

    if selected_pairs.empty:
        logger.error("no pairs survived selection — loosen criteria in config.py")
        sys.exit(1)

    pairs_path = os.path.join(config.results_dir, "selected_pairs.csv")
    selected_pairs.to_csv(pairs_path, index=False)
    logger.info("selected %d pairs -> %s", len(selected_pairs), pairs_path)

    top_pairs = selected_pairs.head(config.top_n_pairs)

    # --- regime detector (fitted on full history for IS backtest) ---
    regime_det = regime_detector.fit(
        market_returns=market_returns,
        vix_series=vix_series,
        train_end_date=None,
        vix_threshold=config.vix_threshold,
        hmm_n_states=config.hmm_n_states,
        use_hmm=config.use_hmm,
    )
    full_regime = regime_det.predict(neutral_prices.index)
    logger.info("regime: %.1f%% risk-off days", (1 - full_regime).mean() * 100)

    # --- per-pair kalman → OU → signals → backtest ---
    pair_results: List[backtest.PairResult] = []

    for _, row in top_pairs.iterrows():
        ty = row["ticker_y"]
        tx = row["ticker_x"]

        if ty not in neutral_prices.columns or tx not in neutral_prices.columns:
            logger.warning("skipping %s/%s — ticker missing from neutral prices", ty, tx)
            continue

        try:
            hedge, intercept, spread = kalman_filter.fit(
                y=neutral_prices[ty],
                x=neutral_prices[tx],
                delta=config.kalman_delta,
                vt=config.kalman_vt,
            )
            z = kalman_filter.zscore(spread, window=20)

            ou_params = None
            try:
                ou_params = ou_process.fit(spread.dropna(), method=config.ou_fitting_method)
            except Exception as ou_exc:
                logger.warning("OU fit failed for %s/%s: %s — using fixed z-thresholds", ty, tx, ou_exc)

            signals = strategy.generate_signals(
                zscore=z,
                regime=full_regime,
                entry_z=config.entry_z,
                exit_z=config.exit_z,
                stop_z=config.stop_z,
                ou_params=ou_params,
                use_ou_thresholds=config.use_ou_thresholds,
                sharpe_target=config.sharpe_target,
                spread=spread,
            )

            ret_y = neutral_returns[ty] if ty in neutral_returns.columns else (
                data_loader.log_returns(stock_prices[[ty]]).squeeze()
            )
            ret_x = neutral_returns[tx] if tx in neutral_returns.columns else (
                data_loader.log_returns(stock_prices[[tx]]).squeeze()
            )

            pr = backtest.run_pair_backtest(
                ticker_y=ty, ticker_x=tx,
                returns_y=ret_y, returns_x=ret_x,
                signals=signals, regime=full_regime,
                hedge_ratio=hedge, spread=spread, zscore=z,
                transaction_cost=config.transaction_cost,
                ou_params=ou_params,
            )
            pair_results.append(pr)

        except Exception as exc:
            logger.error("pair %s/%s failed: %s — skipping", ty, tx, exc)

    if not pair_results:
        logger.error("all pairs failed — check input data and config")
        sys.exit(1)

    portfolio = backtest.run_portfolio_backtest(
        pair_results=pair_results,
        spy_returns=market_returns,
        initial_capital=config.initial_capital,
        sizing_method=config.sizing_method,
        max_pair_weight=config.max_pair_weight,
    )

    elapsed = time.perf_counter() - t0
    logger.info("pipeline done in %.1fs — beta=%.3f, corr_spy=%.3f",
                elapsed, portfolio.portfolio_beta, portfolio.portfolio_correlation_to_spy)

    return portfolio


def _run_sensitivity(
    config: Config,
    neutral_prices: pd.DataFrame,
    neutral_returns: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    market_returns: pd.Series,
    vix_series: pd.Series,
) -> Dict[str, np.ndarray]:
    """
    2D grid search over entry_z × max_half_life.

    For each cell, filter pairs by max_hl and rerun backtest with that entry_z.
    Returns sharpe, cagr, max_dd as 2D numpy arrays.
    """
    entry_z_vals = config.sensitivity_entry_z
    max_hl_vals = config.sensitivity_max_hl
    n_ez = len(entry_z_vals)
    n_hl = len(max_hl_vals)

    sharpe_grid = np.full((n_ez, n_hl), np.nan)
    cagr_grid   = np.full((n_ez, n_hl), np.nan)
    maxdd_grid  = np.full((n_ez, n_hl), np.nan)

    logger.info("sensitivity sweep: %d × %d = %d combos", n_ez, n_hl, n_ez * n_hl)

    regime_det = regime_detector.fit(
        market_returns=market_returns,
        vix_series=vix_series,
        vix_threshold=config.vix_threshold,
        hmm_n_states=config.hmm_n_states,
        use_hmm=config.use_hmm,
    )
    full_regime = regime_det.predict(neutral_prices.index)

    for i, ez in enumerate(entry_z_vals):
        for j, max_hl in enumerate(max_hl_vals):
            try:
                pairs_sub = selected_pairs[selected_pairs["half_life"] <= max_hl].head(config.top_n_pairs)
                if pairs_sub.empty:
                    continue

                pair_results_sub: List[backtest.PairResult] = []
                for _, row in pairs_sub.iterrows():
                    ty, tx = row["ticker_y"], row["ticker_x"]
                    if ty not in neutral_prices.columns or tx not in neutral_prices.columns:
                        continue
                    try:
                        hedge, _, spread = kalman_filter.fit(
                            y=neutral_prices[ty], x=neutral_prices[tx],
                            delta=config.kalman_delta, vt=config.kalman_vt,
                        )
                        z = kalman_filter.zscore(spread, window=20)

                        ou_p = None
                        try:
                            ou_p = ou_process.fit(spread.dropna(), method=config.ou_fitting_method)
                        except Exception:
                            pass

                        # use fixed thresholds for a clean apples-to-apples grid sweep
                        sigs = strategy.generate_signals(
                            zscore=z, regime=full_regime,
                            entry_z=ez, exit_z=max(ez * 0.25, config.exit_z),
                            stop_z=config.stop_z, ou_params=ou_p,
                            use_ou_thresholds=False,
                            sharpe_target=config.sharpe_target, spread=spread,
                        )
                        ret_y = neutral_returns[ty] if ty in neutral_returns.columns else (
                            data_loader.log_returns(neutral_prices[[ty]]).squeeze()
                        )
                        ret_x = neutral_returns[tx] if tx in neutral_returns.columns else (
                            data_loader.log_returns(neutral_prices[[tx]]).squeeze()
                        )
                        pr = backtest.run_pair_backtest(
                            ticker_y=ty, ticker_x=tx,
                            returns_y=ret_y, returns_x=ret_x,
                            signals=sigs, regime=full_regime,
                            hedge_ratio=hedge, spread=spread, zscore=z,
                            transaction_cost=config.transaction_cost,
                        )
                        pair_results_sub.append(pr)
                    except Exception:
                        continue

                if not pair_results_sub:
                    continue

                port_sub = backtest.run_portfolio_backtest(
                    pair_results=pair_results_sub,
                    spy_returns=market_returns,
                    initial_capital=config.initial_capital,
                    sizing_method=config.sizing_method,
                    max_pair_weight=config.max_pair_weight,
                )

                sharpe_grid[i, j] = analytics.sharpe_ratio(port_sub.returns)
                cagr_grid[i, j]   = analytics.cagr_pct(port_sub.equity_curve)
                maxdd_grid[i, j]  = analytics.max_drawdown_pct(port_sub.equity_curve)

            except Exception as exc:
                logger.debug("sensitivity (%.1f, %d) failed: %s", ez, max_hl, exc)

    return {"sharpe": sharpe_grid, "cagr": cagr_grid, "max_dd": maxdd_grid}


def _print_tearsheet(ts: dict) -> None:
    """Print the tearsheet to stdout."""
    w = 52
    bar = "=" * w
    sep = "-" * w

    def _row(label: str, value: str) -> str:
        return f"  {label:<32}  {value:>14}"

    trade = ts.get("trade_stats", {})

    lines = [
        "",
        bar,
        "  STATISTICAL ARBITRAGE -- TEARSHEET".center(w),
        bar,
        _row("Total Return",          f"{ts.get('total_return_pct', 0):.2f}%"),
        _row("CAGR",                  f"{ts.get('cagr_pct', 0):.2f}%"),
        _row("Annualised Volatility", f"{ts.get('annualised_vol_pct', 0):.2f}%"),
        sep,
        _row("Sharpe Ratio",          f"{ts.get('sharpe_ratio', 0):.3f}"),
        _row("Sortino Ratio",         f"{ts.get('sortino_ratio', 0):.3f}"),
        _row("Calmar Ratio",          f"{ts.get('calmar_ratio', 0):.3f}"),
        _row("Omega Ratio",           f"{ts.get('omega_ratio', 0):.3f}"),
        sep,
        _row("Max Drawdown",          f"{ts.get('max_drawdown_pct', 0):.2f}%"),
        _row("Avg Drawdown",          f"{ts.get('avg_drawdown_pct', 0):.2f}%"),
        sep,
        _row("VaR 95%",               f"{ts.get('var_95_pct', 0):.3f}%"),
        _row("CVaR 95%",              f"{ts.get('cvar_95_pct', 0):.3f}%"),
        _row("Skewness",              f"{ts.get('skewness', 0):.4f}"),
        _row("Kurtosis (excess)",     f"{ts.get('excess_kurtosis', 0):.4f}"),
        sep,
        _row("Beta to SPY",           f"{ts.get('portfolio_beta', 0):.4f}"),
        _row("Correlation to SPY",    f"{ts.get('portfolio_corr_to_spy', 0):.4f}"),
        sep,
        _row("Number of Trades",      f"{trade.get('n_trades', 0)}"),
        _row("Win Rate",              f"{trade.get('win_rate', 0)*100:.1f}%"),
        _row("Profit Factor",         f"{trade.get('profit_factor', 0):.3f}"),
        _row("Avg Duration",          f"{trade.get('avg_duration_days', 0):.1f} days"),
        _row("Stop Rate",             f"{trade.get('stop_rate', 0)*100:.1f}%"),
        _row("Expectancy",            f"{trade.get('expectancy_pct', 0):.3f}%"),
        bar,
        "",
    ]
    for line in lines:
        print(line)


def main() -> None:
    args = _parse_args()
    config = Config()

    if args.top_n is not None:
        config.top_n_pairs = args.top_n
        logger.info("top_n_pairs overridden to %d via CLI", config.top_n_pairs)

    os.makedirs(config.results_dir, exist_ok=True)

    portfolio = run_pipeline(config)

    # reload prices for SPY equity curve (already cached so this is instant)
    all_tickers = config.tickers + [config.market_ticker]
    prices = data_loader.load(
        tickers=all_tickers, start=config.start_date, end=config.end_date, cache_dir=".cache"
    )
    spy_prices = prices[config.market_ticker]
    spy_equity = spy_prices / spy_prices.iloc[0]
    stock_prices = prices.drop(columns=[config.market_ticker], errors="ignore")

    spy_ret = data_loader.log_returns(prices)
    market_returns = spy_ret[config.market_ticker].dropna()

    ts = analytics.tearsheet(portfolio, benchmark_returns=market_returns)
    _print_tearsheet(ts)

    ts_path = os.path.join(config.results_dir, "tearsheet.json")
    with open(ts_path, "w") as f:
        json.dump(ts, f, indent=2, default=str)
    logger.info("tearsheet saved to %s", ts_path)

    if args.plot:
        charts_dir = os.path.join(config.results_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        logger.info("saving PNG charts to %s/", charts_dir)

        port_eq = portfolio.equity_curve
        spy_eq = spy_equity.reindex(port_eq.index, method="ffill")

        def _save_png(fig, filename: str) -> None:
            path = os.path.join(charts_dir, filename)
            try:
                fig.write_image(path, width=1400, height=700, scale=2)
                logger.info("saved %s", path)
            except Exception as exc:
                logger.warning("failed to save %s: %s -- skipping", filename, exc)

        _save_png(visualisation.equity_curve(port_eq, spy_eq), "equity_curve.png")
        _save_png(visualisation.drawdown_chart(port_eq), "drawdown.png")

        monthly = analytics.monthly_returns_table(port_eq)
        if not monthly.empty:
            _save_png(visualisation.monthly_returns_heatmap(monthly), "monthly_returns.png")
        else:
            logger.warning("monthly_returns.png skipped -- no monthly data")

        _save_png(visualisation.return_distribution(portfolio.returns), "return_distribution.png")

        # correlation heatmap of neutral returns
        try:
            neutral_p2, neutral_r2, _ = factor_neutralisation.neutralise(
                prices=stock_prices,
                returns=data_loader.log_returns(stock_prices),
                market_returns=market_returns,
                sector_map=config.sector_map,
                neutralise_market=config.neutralise_market,
                neutralise_sector=config.neutralise_sector,
            )
            _save_png(visualisation.correlation_heatmap(neutral_r2), "correlation_heatmap.png")
        except Exception as exc:
            logger.warning("correlation_heatmap.png failed: %s -- skipping", exc)

        # pair dashboard for the first pair
        if portfolio.pair_results:
            pr0 = portfolio.pair_results[0]
            _save_png(
                visualisation.pair_dashboard(pr0, entry_z=config.entry_z, exit_z=config.exit_z, stop_z=config.stop_z),
                "pair_dashboard.png",
            )

        logger.info("charts saved")

    if args.walk_forward:
        logger.info("starting walk-forward validation")

        try:
            vix_df = data_loader.load(
                tickers=["^VIX"], start=config.start_date, end=config.end_date, cache_dir=".cache"
            )
            vix_series = vix_df["^VIX"].dropna()
        except Exception:
            vix_series = pd.Series(20.0, index=market_returns.index, name="^VIX")

        stock_prices = prices.drop(columns=[config.market_ticker], errors="ignore")

        wf_result = walk_forward.run_walk_forward(
            prices=stock_prices,
            spy_prices=spy_prices,
            vix_prices=vix_series,
            config=config,
        )

        # compute OOS trade stats by aggregating across all fold pair results
        oos_all_trades = [
            t
            for fold in wf_result.fold_results
            for pr in fold.portfolio_result.pair_results
            for t in pr.trades
        ]
        oos_trade_stats = analytics.trade_metrics(oos_all_trades)

        # compute OOS beta via portfolio_beta (last fold that had pairs)
        oos_betas = [f.portfolio_result.portfolio_beta for f in wf_result.fold_results if f.n_pairs > 0]
        oos_beta_mean = float(np.mean(oos_betas)) if oos_betas else 0.0

        oos_calmar = (
            wf_result.oos_cagr_pct / abs(wf_result.oos_max_drawdown_pct)
            if wf_result.oos_max_drawdown_pct != 0.0
            else 0.0
        )

        is_sharpe = ts.get("sharpe_ratio", 0.0)
        print("\n=== Walk-Forward OOS Results ===")
        print(f"  OOS Sharpe:        {wf_result.oos_sharpe:.3f}")
        print(f"  OOS CAGR:          {wf_result.oos_cagr_pct:.2f}%")
        print(f"  OOS Max Drawdown:  {wf_result.oos_max_drawdown_pct:.2f}%")
        print(f"  OOS Calmar:        {oos_calmar:.3f}")
        print(f"  OOS Win Rate:      {oos_trade_stats.get('win_rate', 0)*100:.1f}%")
        print(f"  OOS Profit Factor: {oos_trade_stats.get('profit_factor', 0):.3f}")
        print(f"  OOS Beta to SPY:   {oos_beta_mean:.4f}")
        print(f"  In-Sample Sharpe:  {is_sharpe:.3f}")
        print(f"  Folds completed:   {wf_result.n_folds}")
        print()

        # save OOS summary to a json for the README
        oos_summary = {
            "oos_sharpe": wf_result.oos_sharpe,
            "oos_cagr_pct": wf_result.oos_cagr_pct,
            "oos_max_drawdown_pct": wf_result.oos_max_drawdown_pct,
            "oos_calmar": oos_calmar,
            "oos_win_rate": oos_trade_stats.get("win_rate", 0),
            "oos_profit_factor": oos_trade_stats.get("profit_factor", 0),
            "oos_beta": oos_beta_mean,
            "n_folds": wf_result.n_folds,
        }
        oos_path = os.path.join(config.results_dir, "oos_summary.json")
        with open(oos_path, "w") as f:
            json.dump(oos_summary, f, indent=2, default=str)
        logger.info("OOS summary saved to %s", oos_path)

        fold_data = pd.DataFrame([
            {
                "fold": r.fold_index,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                "oos_sharpe": r.oos_sharpe,
                "n_pairs": r.n_pairs,
            }
            for r in wf_result.fold_results
        ])
        fold_path = os.path.join(config.results_dir, "fold_summary.csv")
        fold_data.to_csv(fold_path, index=False)
        logger.info("fold summary saved to %s", fold_path)

        if args.plot and not fold_data.empty:
            charts_dir = os.path.join(config.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            try:
                visualisation.walk_forward_chart(wf_result.oos_equity, fold_data).write_image(
                    os.path.join(charts_dir, "walk_forward.png"), width=1400, height=700, scale=2
                )
                logger.info("walk-forward chart saved")
            except Exception as exc:
                logger.warning("walk_forward.png failed: %s -- skipping", exc)

    if args.sensitivity:
        logger.info("starting sensitivity sweep")

        stock_prices = prices.drop(columns=[config.market_ticker], errors="ignore")
        stock_returns = data_loader.log_returns(stock_prices)

        try:
            vix_sens = data_loader.load(
                tickers=["^VIX"], start=config.start_date, end=config.end_date, cache_dir=".cache"
            )["^VIX"].dropna()
        except Exception:
            vix_sens = pd.Series(20.0, index=market_returns.index, name="^VIX")

        # pair selection with the widest max_hl so all grid cells have data to work with
        neutral_p, neutral_r, _ = factor_neutralisation.neutralise(
            prices=stock_prices, returns=stock_returns, market_returns=market_returns,
            sector_map=config.sector_map, neutralise_market=config.neutralise_market,
            neutralise_sector=config.neutralise_sector,
        )
        pairs_wide = pair_selection.select_pairs(
            neutral_prices=neutral_p, neutral_returns=neutral_r,
            sector_map=config.sector_map, min_correlation=config.min_correlation,
            coint_pvalue=config.coint_pvalue, min_half_life=config.min_half_life,
            max_half_life=max(config.sensitivity_max_hl),
            rolling_coint_window=config.rolling_coint_window,
            max_pairs_per_sector=config.max_pairs_per_sector,
        )

        sens_grid = _run_sensitivity(
            config=config,
            neutral_prices=neutral_p, neutral_returns=neutral_r,
            selected_pairs=pairs_wide, market_returns=market_returns,
            vix_series=vix_sens,
        )

        if args.plot:
            charts_dir = os.path.join(config.results_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            try:
                visualisation.sensitivity_heatmap(
                    sens_grid,
                    param1_values=config.sensitivity_entry_z,
                    param2_values=config.sensitivity_max_hl,
                ).write_image(
                    os.path.join(charts_dir, "sensitivity_heatmap.png"), width=1200, height=800, scale=2
                )
                logger.info("sensitivity heatmap saved")
            except Exception as exc:
                logger.warning("sensitivity_heatmap.png failed: %s -- skipping", exc)

        print("\n=== Sensitivity: Sharpe Grid (entry_z rows × max_hl cols) ===")
        sharpe_df = pd.DataFrame(
            sens_grid["sharpe"],
            index=[f"entry_z={ez}" for ez in config.sensitivity_entry_z],
            columns=[f"max_hl={h}" for h in config.sensitivity_max_hl],
        )
        print(sharpe_df.round(3).to_string())
        print()

    logger.info("all done — results in %s/", config.results_dir)


if __name__ == "__main__":
    # multiprocessing on Windows requires this guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
