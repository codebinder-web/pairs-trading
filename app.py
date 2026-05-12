"""
app.py
======
Multi-tab Streamlit dashboard for the statistical arbitrage pairs trading system.

Tabs
----
📊 Overview       — KPIs, equity curve, drawdown, monthly heatmap, return dist.
🔍 Pair Selection — Selected pairs table, correlation heatmap, coint scatter.
🔬 Pair Deep-Dive — Per-pair dashboard, OU diagnostics, trade log.
📅 Walk-Forward   — OOS equity curve, fold summary, robustness assessment.
🎛️ Sensitivity    — 2D sensitivity heatmap.
⚠️ Risk           — Beta/correlation, sector exposure, VaR/CVaR, factor betas.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pairs Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for dark theme and KPI cards
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        margin: 4px;
    }
    .kpi-label { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 24px; font-weight: 700; color: #e0e6f0; margin-top: 4px; }
    .kpi-value.positive { color: #00d084; }
    .kpi-value.negative { color: #ff4444; }
    .kpi-value.neutral  { color: #4fc3f7; }
    .section-header { color: #8892a4; font-size: 12px; text-transform: uppercase;
                      letter-spacing: 1.5px; border-bottom: 1px solid #2d3250;
                      padding-bottom: 4px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
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
from main import run_pipeline, run_sensitivity


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _state(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

def _build_sidebar() -> Config:
    """Render sidebar controls and return a Config object."""
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("Adjust parameters, then click **Run Backtest**.")

        with st.expander("🗓️ Universe & Dates", expanded=True):
            start = st.text_input("Start Date", "2018-01-01")
            end = st.text_input("End Date", "2024-12-31")

        with st.expander("🔍 Pair Selection"):
            min_corr = st.slider("Min Correlation", 0.50, 0.95, 0.70, 0.05)
            coint_p = st.slider("EG p-value threshold", 0.01, 0.10, 0.05, 0.01)
            min_hl = st.slider("Min Half-Life (days)", 2, 20, 5, 1)
            max_hl = st.slider("Max Half-Life (days)", 20, 120, 60, 5)
            max_ps = st.slider("Max Pairs / Sector", 1, 5, 2, 1)

        with st.expander("📡 Signal Generation"):
            entry_z = st.slider("Entry Z-Score", 1.0, 4.0, 2.0, 0.25)
            exit_z = st.slider("Exit Z-Score", 0.0, 1.5, 0.5, 0.25)
            stop_z = st.slider("Stop-Loss Z-Score", 2.5, 6.0, 4.0, 0.25)
            use_ou = st.checkbox("Use OU-optimal thresholds", value=True)
            ou_method = st.selectbox("OU fitting method", ["mle", "ols"])

        with st.expander("💰 Position Sizing"):
            sizing = st.selectbox("Sizing Method", ["half_kelly", "kelly", "equal"])
            max_pw = st.slider("Max Pair Weight", 0.05, 0.50, 0.15, 0.05)
            top_n = st.slider("Top N Pairs", 2, 20, 10, 1)
            tc = st.number_input("Transaction Cost (bps)", 0, 50, 10, 1) / 10000.0

        with st.expander("🌊 Regime Detection"):
            vix_thresh = st.slider("VIX Threshold", 15.0, 40.0, 25.0, 1.0)
            use_hmm = st.checkbox("Use HMM Regime Detector", value=True)

        with st.expander("💼 Portfolio"):
            init_cap = st.number_input("Initial Capital ($)", 10000, 10000000, 100000, 10000)

        st.markdown("---")
        run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)
        wf_btn = st.button("▶  Run Walk-Forward", use_container_width=True)
        sens_btn = st.button("▶  Run Sensitivity", use_container_width=True)

        st.session_state["run_btn"] = run_btn
        st.session_state["wf_btn"] = wf_btn
        st.session_state["sens_btn"] = sens_btn

    return Config(
        start_date=start,
        end_date=end,
        min_correlation=min_corr,
        coint_pvalue=coint_p,
        min_half_life=min_hl,
        max_half_life=max_hl,
        max_pairs_per_sector=max_ps,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        use_ou_thresholds=use_ou,
        ou_fitting_method=ou_method,
        sizing_method=sizing,
        max_pair_weight=max_pw,
        top_n_pairs=top_n,
        transaction_cost=tc,
        vix_threshold=vix_thresh,
        use_hmm=use_hmm,
        initial_capital=float(init_cap),
    )


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------

def _kpi(label: str, value: str, positive: Optional[bool] = None) -> str:
    css_class = "neutral"
    if positive is True:
        css_class = "positive"
    elif positive is False:
        css_class = "negative"
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {css_class}">{value}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def _tab_overview(ts: dict, portfolio: backtest.PortfolioResult, spy_equity: pd.Series) -> None:
    st.subheader("Portfolio Performance Overview")

    # KPI Row
    kpis = [
        ("Total Return",   f"{ts['total_return_pct']:.1f}%",    ts['total_return_pct'] > 0),
        ("CAGR",           f"{ts['cagr_pct']:.1f}%",             ts['cagr_pct'] > 0),
        ("Sharpe Ratio",   f"{ts['sharpe_ratio']:.2f}",          ts['sharpe_ratio'] > 0),
        ("Sortino Ratio",  f"{ts['sortino_ratio']:.2f}",         ts['sortino_ratio'] > 0),
        ("Calmar Ratio",   f"{ts['calmar_ratio']:.2f}",          ts['calmar_ratio'] > 0),
        ("Max Drawdown",   f"{ts['max_drawdown_pct']:.1f}%",     False),
        ("Win Rate",       f"{ts['win_rate']*100:.1f}%",         ts['win_rate'] > 0.5),
        ("Beta to SPY",    f"{ts['beta_to_spy']:.3f}",           abs(ts['beta_to_spy']) < 0.1),
    ]

    cols = st.columns(8)
    for col, (label, value, is_positive) in zip(cols, kpis):
        with col:
            st.markdown(_kpi(label, value, is_positive), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            visualisation.equity_curve(portfolio.equity_curve, spy_equity),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            visualisation.drawdown_chart(portfolio.equity_curve),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        monthly = analytics.monthly_returns_table(portfolio.equity_curve)
        if not monthly.empty:
            st.plotly_chart(
                visualisation.monthly_returns_heatmap(monthly),
                use_container_width=True,
            )
    with col4:
        st.plotly_chart(
            visualisation.return_distribution(portfolio.returns),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab: Pair Selection
# ---------------------------------------------------------------------------

def _tab_pair_selection(
    selected_pairs: pd.DataFrame,
    neutral_returns: pd.DataFrame,
) -> None:
    st.subheader("Selected Pairs")

    if selected_pairs.empty:
        st.warning("No pairs selected. Loosen selection criteria.")
        return

    # Conditional formatting on score column
    score_col = "score" if "score" in selected_pairs.columns else None
    display_cols = [c for c in [
        "ticker_y", "ticker_x", "sector_y", "sector_x",
        "correlation", "eg_pvalue", "johansen_reject",
        "hedge_ratio", "half_life", "spread_std", "score",
    ] if c in selected_pairs.columns]

    st.dataframe(
        selected_pairs[display_cols].style.format({
            "correlation": "{:.3f}",
            "eg_pvalue": "{:.4f}",
            "hedge_ratio": "{:.3f}",
            "half_life": "{:.1f}",
            "spread_std": "{:.4f}",
            "score": "{:.4f}",
        }).background_gradient(subset=["score"] if "score" in display_cols else []),
        use_container_width=True,
        height=min(400, 40 + 36 * len(selected_pairs)),
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            visualisation.correlation_heatmap(neutral_returns),
            use_container_width=True,
        )

    with col2:
        if len(selected_pairs) > 0:
            pair_idx = st.selectbox(
                "Select pair for cointegration scatter",
                range(len(selected_pairs)),
                format_func=lambda i: f"{selected_pairs.iloc[i]['ticker_y']} / {selected_pairs.iloc[i]['ticker_x']}",
                key="coint_scatter_pair",
            )
            row = selected_pairs.iloc[pair_idx]
            ty, tx = row["ticker_y"], row["ticker_x"]
            if ty in neutral_returns.columns and tx in neutral_returns.columns:
                # Reconstruct neutral prices from returns
                neutral_prices_approx = 100.0 * np.exp(neutral_returns[[ty, tx]].cumsum())
                st.plotly_chart(
                    visualisation.cointegration_scatter(
                        prices_y=neutral_prices_approx[ty],
                        prices_x=neutral_prices_approx[tx],
                        hedge_ratio=float(row.get("hedge_ratio", 1.0)),
                        ticker_y=ty,
                        ticker_x=tx,
                    ),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Tab: Pair Deep-Dive
# ---------------------------------------------------------------------------

def _tab_pair_deep_dive(
    portfolio: backtest.PortfolioResult,
    config: Config,
) -> None:
    st.subheader("Pair Deep-Dive")

    if not portfolio.pair_results:
        st.warning("No pair results available. Run backtest first.")
        return

    pair_labels = [
        f"{pr.ticker_y} / {pr.ticker_x}"
        for pr in portfolio.pair_results
    ]
    selected_label = st.selectbox("Choose a pair", pair_labels)
    pair_idx = pair_labels.index(selected_label)
    pr = portfolio.pair_results[pair_idx]

    # Pair dashboard chart
    st.plotly_chart(
        visualisation.pair_dashboard(
            pr,
            entry_z=config.entry_z,
            exit_z=config.exit_z,
            stop_z=config.stop_z,
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    # OU parameters
    with col1:
        st.markdown('<p class="section-header">OU Process Parameters</p>', unsafe_allow_html=True)
        if pr.ou_params is not None:
            ou = pr.ou_params
            ou_data = {
                "Parameter": ["κ (speed)", "μ (mean)", "σ (diffusion)", "σ_eq (equilibrium)", "Half-life"],
                "Value": [
                    f"{ou.kappa:.5f}",
                    f"{ou.mu:.5f}",
                    f"{ou.sigma:.5f}",
                    f"{ou.sigma_eq:.5f}",
                    f"{ou.half_life:.1f} days",
                ],
            }
            st.table(pd.DataFrame(ou_data))
        else:
            st.info("OU parameters not available for this pair.")

    # Per-pair KPIs
    with col2:
        st.markdown('<p class="section-header">Pair Performance</p>', unsafe_allow_html=True)
        pair_ts = analytics.pair_tearsheet(pr)
        kpi_keys = [
            ("Total Return", "total_return_pct", "%"),
            ("Sharpe Ratio", "sharpe_ratio", ""),
            ("Max Drawdown", "max_drawdown_pct", "%"),
            ("Win Rate", "win_rate", " (×100 = %)"),
            ("N Trades", "n_trades", ""),
            ("Avg Duration", "avg_duration_days", " days"),
            ("Sizing Weight", "sizing_weight", ""),
        ]
        kpi_df = pd.DataFrame(
            [(label, pair_ts.get(key, "—")) for label, key, _ in kpi_keys],
            columns=["Metric", "Value"],
        )
        st.table(kpi_df)

    # Trade log
    if pr.trades:
        st.markdown('<p class="section-header">Trade Log</p>', unsafe_allow_html=True)
        trades_df = pd.DataFrame([{
            "Entry": t.entry_date.date(),
            "Exit": t.exit_date.date(),
            "Direction": "Long Y" if t.direction == 1 else "Short Y",
            "PnL %": f"{t.pnl_pct:.3f}",
            "Duration": t.duration_days,
            "Entry Z": f"{t.entry_z:.2f}",
            "Exit Z": f"{t.exit_z:.2f}",
            "Stop": "✓" if t.stop_triggered else "",
        } for t in pr.trades])
        st.dataframe(trades_df, use_container_width=True, height=300)
    else:
        st.info("No completed trades for this pair.")


# ---------------------------------------------------------------------------
# Tab: Walk-Forward
# ---------------------------------------------------------------------------

def _tab_walk_forward(wf_result) -> None:
    st.subheader("Out-of-Sample Walk-Forward Validation")

    if wf_result is None:
        st.info("Click **Run Walk-Forward** in the sidebar to execute OOS validation.")
        return

    # Robustness banner
    robustness_msg = (
        "✅ **Robust** — OOS Sharpe ≥ 0.5 × In-Sample Sharpe"
        if wf_result.is_robust
        else "⚠️ **Possible Overfit** — OOS Sharpe < 0.5 × In-Sample Sharpe"
    )
    st.success(robustness_msg) if wf_result.is_robust else st.warning(robustness_msg)

    col1, col2, col3 = st.columns(3)
    col1.metric("OOS Sharpe", f"{wf_result.oos_sharpe:.3f}")
    col2.metric("IS Sharpe", f"{wf_result.in_sample_sharpe:.3f}")
    col3.metric(
        "OOS Total Return",
        f"{analytics.total_return_pct(wf_result.oos_equity):.1f}%"
    )

    # Walk-forward chart
    if not wf_result.fold_summary.empty:
        st.plotly_chart(
            visualisation.walk_forward_chart(wf_result.oos_equity, wf_result.fold_summary),
            use_container_width=True,
        )

        st.markdown('<p class="section-header">Fold Summary</p>', unsafe_allow_html=True)
        st.dataframe(wf_result.fold_summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Sensitivity
# ---------------------------------------------------------------------------

def _tab_sensitivity(
    sens_grid: Optional[Dict],
    config: Config,
) -> None:
    st.subheader("Sensitivity Analysis")

    if sens_grid is None:
        st.info("Click **Run Sensitivity** in the sidebar to run the parameter grid search.")
        return

    fig = visualisation.sensitivity_heatmap(
        sens_grid,
        param1_values=config.sensitivity_entry_z,
        param2_values=config.sensitivity_max_hl,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpret stable region
    sharpe_arr = sens_grid.get("sharpe", np.zeros((1, 1)))
    stable_mask = sharpe_arr >= 1.0
    n_stable = int(np.sum(stable_mask))
    total = sharpe_arr.size

    if n_stable > 0:
        st.success(
            f"✅ **{n_stable}/{total}** parameter combinations achieve Sharpe ≥ 1.0 — "
            "the strategy is robust across this region."
        )
    else:
        st.warning(
            "No parameter combination achieves Sharpe ≥ 1.0 in this grid. "
            "Consider widening the search range or reviewing the strategy."
        )


# ---------------------------------------------------------------------------
# Tab: Risk
# ---------------------------------------------------------------------------

def _tab_risk(
    ts: dict,
    portfolio: backtest.PortfolioResult,
    betas_df: pd.DataFrame,
    config: Config,
) -> None:
    st.subheader("Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    beta = ts.get("beta_to_spy", 0)
    col1.metric("Beta to SPY", f"{beta:.4f}",
                help="Near-zero confirms market neutrality — the whole point of pairs trading.")
    col2.metric("Correlation to SPY", f"{ts.get('correlation_to_spy', 0):.4f}")
    col3.metric("VaR 95%", f"{ts.get('var_95_pct', 0):.3f}%")
    col4.metric("CVaR 95%", f"{ts.get('cvar_95_pct', 0):.3f}%")

    if abs(beta) < 0.1:
        st.success(
            f"✅ Portfolio beta = {beta:.4f} — near-zero confirms market neutrality."
        )
    else:
        st.warning(
            f"⚠️ Portfolio beta = {beta:.4f} — this is higher than expected for a "
            "pairs strategy. Review factor neutralisation."
        )

    # VaR/CVaR visualisation
    st.plotly_chart(
        visualisation.return_distribution(portfolio.returns),
        use_container_width=True,
    )

    # Sector exposure
    if portfolio.pair_results:
        sector_weights: Dict[str, float] = {}
        for pr in portfolio.pair_results:
            sector = config.sector_map.get(pr.ticker_y, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + pr.sizing_weight

        import plotly.graph_objects as go
        fig_sector = go.Figure(go.Bar(
            x=list(sector_weights.keys()),
            y=list(sector_weights.values()),
            marker=dict(color="#4fc3f7"),
        ))
        fig_sector.update_layout(
            template="plotly_dark",
            title="Sector Exposure (by sizing weight)",
            xaxis_title="Sector",
            yaxis_title="Aggregate Weight",
            height=350,
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    # Factor betas table
    if not betas_df.empty:
        st.markdown('<p class="section-header">Factor Betas (β_mkt and β_sec per ticker)</p>',
                    unsafe_allow_html=True)
        st.dataframe(
            betas_df.style.background_gradient(
                subset=["beta_mkt", "beta_sec"] if "beta_mkt" in betas_df.columns else [],
                cmap="RdYlGn",
            ).format("{:.4f}"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📈 Statistical Arbitrage Pairs Trading System")
    st.caption(
        "Factor-neutralised cointegration | Kalman filter hedge ratio | "
        "OU-optimal thresholds | HMM regime detection | Walk-forward OOS validation"
    )

    config = _build_sidebar()

    # Tab layout
    tabs = st.tabs([
        "📊 Overview",
        "🔍 Pair Selection",
        "🔬 Pair Deep-Dive",
        "📅 Walk-Forward",
        "🎛️ Sensitivity",
        "⚠️ Risk",
    ])

    # ------------------------------------------------------------------
    # Run main backtest
    # ------------------------------------------------------------------
    if st.session_state.get("run_btn"):
        with st.spinner("Running pipeline… (data download, factor neutralisation, pair selection, Kalman, backtest)"):
            try:
                portfolio = run_pipeline(config)

                # Load supporting data for analytics
                all_tickers = config.tickers + [config.market_ticker]
                prices = data_loader.load(
                    tickers=all_tickers,
                    start=config.start_date,
                    end=config.end_date,
                )
                spy_prices = prices[config.market_ticker]
                spy_equity = spy_prices / spy_prices.iloc[0]
                spy_returns = data_loader.log_returns(prices[[config.market_ticker]]).squeeze()

                # Factor neutralisation for tab use
                stock_prices = prices.drop(columns=[config.market_ticker], errors="ignore")
                stock_returns = data_loader.log_returns(stock_prices)
                neutral_prices, neutral_returns, betas_df = factor_neutralisation.neutralise(
                    prices=stock_prices,
                    returns=stock_returns,
                    market_returns=spy_returns,
                    sector_map=config.sector_map,
                    neutralise_market=config.neutralise_market,
                    neutralise_sector=config.neutralise_sector,
                )

                selected_pairs_path = os.path.join(config.results_dir, "selected_pairs.csv")
                if os.path.exists(selected_pairs_path):
                    selected_pairs = pd.read_csv(selected_pairs_path)
                else:
                    selected_pairs = pd.DataFrame()

                ts = analytics.tearsheet(
                    portfolio_equity=portfolio.equity_curve,
                    spy_equity=spy_equity.reindex(portfolio.equity_curve.index, method="ffill"),
                    pair_results=portfolio.pair_results,
                    initial_capital=config.initial_capital,
                )

                # Persist to session state
                st.session_state["portfolio"] = portfolio
                st.session_state["ts"] = ts
                st.session_state["spy_equity"] = spy_equity
                st.session_state["neutral_returns"] = neutral_returns
                st.session_state["selected_pairs"] = selected_pairs
                st.session_state["betas_df"] = betas_df
                st.session_state["config"] = config
                st.session_state["wf_result"] = None
                st.session_state["sens_grid"] = None

                st.success(
                    f"✅ Backtest complete. "
                    f"Sharpe={ts['sharpe_ratio']:.3f} | "
                    f"CAGR={ts['cagr_pct']:.1f}% | "
                    f"MaxDD={ts['max_drawdown_pct']:.1f}%"
                )

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                logger.exception("Pipeline failed:")

    # ------------------------------------------------------------------
    # Run walk-forward
    # ------------------------------------------------------------------
    if st.session_state.get("wf_btn"):
        if "portfolio" not in st.session_state:
            st.warning("Run the main backtest first before walk-forward.")
        else:
            with st.spinner("Running walk-forward validation…"):
                try:
                    cfg = st.session_state.get("config", config)
                    all_tickers = cfg.tickers + [cfg.market_ticker]
                    prices_wf = data_loader.load(
                        tickers=all_tickers,
                        start=cfg.start_date,
                        end=cfg.end_date,
                    )
                    all_ret_wf = data_loader.log_returns(prices_wf)
                    mkt_ret_wf = all_ret_wf[cfg.market_ticker].dropna()
                    stock_ret_wf = all_ret_wf.drop(columns=[cfg.market_ticker], errors="ignore")
                    stock_prices_wf = prices_wf.drop(columns=[cfg.market_ticker], errors="ignore")
                    try:
                        vix_wf = data_loader.load(
                            tickers=["^VIX"],
                            start=cfg.start_date,
                            end=cfg.end_date,
                        )["^VIX"].dropna()
                    except Exception:
                        vix_wf = pd.Series(20.0, index=mkt_ret_wf.index, name="^VIX")

                    is_sharpe = st.session_state["ts"].get("sharpe_ratio", 0.0)
                    wf_result = walk_forward.run_walk_forward(
                        config=cfg,
                        prices=stock_prices_wf,
                        returns=stock_ret_wf,
                        market_prices=prices_wf[cfg.market_ticker],
                        market_returns=mkt_ret_wf,
                        vix_series=vix_wf,
                        in_sample_sharpe=is_sharpe,
                    )
                    st.session_state["wf_result"] = wf_result
                    st.success(
                        f"✅ Walk-forward complete. OOS Sharpe={wf_result.oos_sharpe:.3f}. "
                        f"{'Robust ✓' if wf_result.is_robust else 'Possible overfit ⚠️'}"
                    )
                except Exception as exc:
                    st.error(f"Walk-forward error: {exc}")
                    logger.exception("Walk-forward failed:")

    # ------------------------------------------------------------------
    # Run sensitivity
    # ------------------------------------------------------------------
    if st.session_state.get("sens_btn"):
        with st.spinner("Running sensitivity analysis… (this may take several minutes)"):
            try:
                cfg = st.session_state.get("config", config)
                all_tickers = cfg.tickers + [cfg.market_ticker]
                prices_s = data_loader.load(
                    tickers=all_tickers,
                    start=cfg.start_date,
                    end=cfg.end_date,
                )
                all_ret_s = data_loader.log_returns(prices_s)
                mkt_ret_s = all_ret_s[cfg.market_ticker].dropna()
                stock_ret_s = all_ret_s.drop(columns=[cfg.market_ticker], errors="ignore")
                stock_prices_s = prices_s.drop(columns=[cfg.market_ticker], errors="ignore")
                neu_p, neu_r, _ = factor_neutralisation.neutralise(
                    prices=stock_prices_s,
                    returns=stock_ret_s,
                    market_returns=mkt_ret_s,
                    sector_map=cfg.sector_map,
                    neutralise_market=cfg.neutralise_market,
                    neutralise_sector=cfg.neutralise_sector,
                )
                sel_pairs_s = pair_selection.select_pairs(
                    neutral_prices=neu_p,
                    neutral_returns=neu_r,
                    sector_map=cfg.sector_map,
                    min_correlation=cfg.min_correlation,
                    coint_pvalue=cfg.coint_pvalue,
                    min_half_life=cfg.min_half_life,
                    max_half_life=max(cfg.sensitivity_max_hl),
                    rolling_coint_window=cfg.rolling_coint_window,
                    max_pairs_per_sector=cfg.max_pairs_per_sector,
                )
                try:
                    vix_s = data_loader.load(
                        tickers=["^VIX"],
                        start=cfg.start_date,
                        end=cfg.end_date,
                    )["^VIX"].dropna()
                except Exception:
                    vix_s = pd.Series(20.0, index=mkt_ret_s.index, name="^VIX")

                sens_grid = run_sensitivity(
                    config=cfg,
                    prices=stock_prices_s,
                    neutral_prices=neu_p,
                    neutral_returns=neu_r,
                    market_returns=mkt_ret_s,
                    selected_pairs=sel_pairs_s,
                    vix_series=vix_s,
                )
                st.session_state["sens_grid"] = sens_grid
                st.success("✅ Sensitivity analysis complete.")
            except Exception as exc:
                st.error(f"Sensitivity error: {exc}")
                logger.exception("Sensitivity analysis failed:")

    # ------------------------------------------------------------------
    # Render tabs
    # ------------------------------------------------------------------
    has_results = "ts" in st.session_state and st.session_state["ts"]

    with tabs[0]:  # Overview
        if has_results:
            _tab_overview(
                ts=st.session_state["ts"],
                portfolio=st.session_state["portfolio"],
                spy_equity=st.session_state["spy_equity"],
            )
        else:
            st.info("Run the backtest from the sidebar to see results here.")

    with tabs[1]:  # Pair Selection
        if has_results:
            _tab_pair_selection(
                selected_pairs=st.session_state.get("selected_pairs", pd.DataFrame()),
                neutral_returns=st.session_state.get("neutral_returns", pd.DataFrame()),
            )
        else:
            st.info("Run the backtest to see pair selection results.")

    with tabs[2]:  # Pair Deep-Dive
        if has_results:
            _tab_pair_deep_dive(
                portfolio=st.session_state["portfolio"],
                config=st.session_state.get("config", config),
            )
        else:
            st.info("Run the backtest to explore individual pairs.")

    with tabs[3]:  # Walk-Forward
        _tab_walk_forward(st.session_state.get("wf_result"))

    with tabs[4]:  # Sensitivity
        _tab_sensitivity(
            sens_grid=st.session_state.get("sens_grid"),
            config=st.session_state.get("config", config),
        )

    with tabs[5]:  # Risk
        if has_results:
            _tab_risk(
                ts=st.session_state["ts"],
                portfolio=st.session_state["portfolio"],
                betas_df=st.session_state.get("betas_df", pd.DataFrame()),
                config=st.session_state.get("config", config),
            )
        else:
            st.info("Run the backtest to see risk metrics.")


if __name__ == "__main__":
    main()
