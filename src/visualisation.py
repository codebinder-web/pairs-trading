"""
visualisation.py — Plotly chart builders for the dashboard.

All dark theme (plotly_dark), all return go.Figure, no side effects.
The dashboard/main.py calls these and decides what to do with the output.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.backtest import PairResult

logger = logging.getLogger(__name__)

_TEMPLATE = "plotly_dark"
_GREEN  = "#00d084"
_RED    = "#ff4444"
_BLUE   = "#4fc3f7"
_ORANGE = "#ffb74d"
_GREY   = "rgba(120,120,120,0.3)"


def equity_curve(
    portfolio_equity: pd.Series,
    spy_equity: pd.Series,
    title: str = "Strategy vs SPY Buy-and-Hold",
) -> go.Figure:
    """Dual equity curve — both rebased to 100 at start."""
    port = portfolio_equity * 100
    spy = spy_equity.reindex(portfolio_equity.index, method="ffill") * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port.index, y=port.values,
        name="Strategy",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=spy.index, y=spy.values,
        name="SPY B&H",
        line=dict(color=_ORANGE, width=1.5, dash="dot"),
    ))
    _apply_layout(fig, title=title, yaxis_title="Value (rebased 100)")
    return fig


def drawdown_chart(
    equity: pd.Series,
    title: str = "Underwater Plot (Drawdown from Peak)",
) -> go.Figure:
    """Red underwater plot — how far below the prior high we are at each point."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown %",
        fill="tozeroy",
        fillcolor="rgba(255,68,68,0.25)",
        line=dict(color=_RED, width=1.5),
    ))
    fig.add_hline(y=0, line=dict(color="white", width=0.5, dash="dot"))
    _apply_layout(fig, title=title, yaxis_title="Drawdown (%)")
    return fig


def monthly_returns_heatmap(
    monthly_returns_df: pd.DataFrame,
    title: str = "Monthly Returns (%)",
) -> go.Figure:
    """Calendar heatmap of monthly returns — rows are years, columns are months."""
    if monthly_returns_df.empty:
        return _empty_figure("no monthly return data available")

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_cols = [c for c in month_names if c in monthly_returns_df.columns]
    z = monthly_returns_df[month_cols].values * 100.0
    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str) + "%")

    fig = go.Figure(go.Heatmap(
        z=z,
        x=month_cols,
        y=[str(y) for y in monthly_returns_df.index],
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        showscale=True,
        colorbar=dict(title="Return %"),
    ))
    _apply_layout(fig, title=title, height=40 * len(monthly_returns_df) + 120)
    return fig


def return_distribution(
    returns: pd.Series,
    title: str = "Daily Return Distribution",
) -> go.Figure:
    """Histogram of daily returns with fitted normal and VaR/CVaR lines annotated."""
    r = returns.dropna().values
    if len(r) < 5:
        return _empty_figure("not enough return data")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r * 100.0,
        nbinsx=60,
        name="Daily Returns",
        histnorm="probability density",
        marker=dict(color=_BLUE, opacity=0.7),
    ))

    # overlay fitted normal — mostly useful for seeing how fat the tails are
    mu_fit = float(np.mean(r)) * 100.0
    sigma_fit = float(np.std(r, ddof=1)) * 100.0
    x_grid = np.linspace(mu_fit - 4 * sigma_fit, mu_fit + 4 * sigma_fit, 200)
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=norm.pdf(x_grid, mu_fit, sigma_fit),
        name="Fitted Normal",
        line=dict(color=_ORANGE, width=2),
    ))

    var_5 = float(np.percentile(r, 5)) * 100.0
    cvar_5 = float(np.mean(r[r <= np.percentile(r, 5)])) * 100.0
    fig.add_vline(x=var_5, line=dict(color=_RED, width=1.5, dash="dash"),
                  annotation_text=f"VaR 95%: {var_5:.2f}%", annotation_position="top right")
    fig.add_vline(x=cvar_5, line=dict(color=_RED, width=1, dash="dot"),
                  annotation_text=f"CVaR 95%: {cvar_5:.2f}%", annotation_position="bottom right")

    _apply_layout(fig, title=title, xaxis_title="Daily Return (%)", yaxis_title="Density")
    return fig


def pair_dashboard(
    pair_result: PairResult,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> go.Figure:
    """
    3-row pair dashboard: normalised prices / Kalman spread / z-score with signals.

    Regime risk-off periods are shaded grey. Trade entries and exits are marked
    with triangles on the z-score panel.
    """
    pr = pair_result
    ty, tx = pr.ticker_y, pr.ticker_x

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            f"{ty} vs {tx} — Normalised Prices",
            "Kalman Spread",
            "Z-Score + Trade Signals",
        ],
        row_heights=[0.28, 0.36, 0.36],
    )

    dates = pr.spread.index
    spread_vals = pr.spread.values
    z_vals = pr.zscore.values
    regime_vals = pr.regime.reindex(dates, fill_value=0).values

    # row 2 — spread with OU equilibrium bands if available
    fig.add_trace(go.Scatter(
        x=dates, y=spread_vals,
        name="Kalman Spread",
        line=dict(color=_BLUE, width=1.2),
    ), row=2, col=1)

    if pr.ou_params is not None:
        mu, sig_eq = pr.ou_params.mu, pr.ou_params.sigma_eq
        fig.add_hline(y=mu, line=dict(color="white", width=0.8, dash="dot"), row=2, col=1)
        fig.add_hline(y=mu + 2 * sig_eq, line=dict(color=_GREEN, width=0.8), row=2, col=1)
        fig.add_hline(y=mu - 2 * sig_eq, line=dict(color=_GREEN, width=0.8), row=2, col=1)

    # row 3 — z-score with threshold lines
    fig.add_trace(go.Scatter(
        x=dates, y=z_vals,
        name="Z-Score",
        line=dict(color=_BLUE, width=1.2),
    ), row=3, col=1)

    for level, colour, label in [
        ( entry_z, _GREEN, f"+{entry_z}σ entry"),
        (-entry_z, _GREEN, f"-{entry_z}σ entry"),
        ( exit_z,  _ORANGE, f"+{exit_z}σ exit"),
        (-exit_z,  _ORANGE, f"-{exit_z}σ exit"),
        ( stop_z,  _RED, f"+{stop_z}σ stop"),
        (-stop_z,  _RED, f"-{stop_z}σ stop"),
    ]:
        fig.add_hline(y=level, line=dict(color=colour, width=1, dash="dash"), row=3, col=1)

    # shade grey during regime=0 periods
    _add_regime_shading(fig, dates, regime_vals, row=3)
    _add_trade_markers(fig, pr.trades, row=3)

    fig.update_layout(template=_TEMPLATE, height=700, showlegend=True)
    fig.update_xaxes(showgrid=False)
    return fig


def cointegration_scatter(
    prices_y: pd.Series,
    prices_x: pd.Series,
    hedge_ratio: float,
    ticker_y: str,
    ticker_x: str,
) -> go.Figure:
    """Y vs β·X scatter with OLS regression line — visual check of the cointegrating relationship."""
    common = prices_y.index.intersection(prices_x.index)
    y_vals = prices_y.loc[common].values
    x_vals = prices_x.loc[common].values * hedge_ratio

    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        name=f"{ticker_y} vs β·{ticker_x}",
        marker=dict(color=_BLUE, size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        name=f"OLS fit (β={hedge_ratio:.3f})",
        line=dict(color=_ORANGE, width=2),
    ))
    _apply_layout(
        fig,
        title=f"Cointegration: {ticker_y} vs β·{ticker_x} (β={hedge_ratio:.3f})",
        xaxis_title=f"β·{ticker_x}",
        yaxis_title=ticker_y,
    )
    return fig


def correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Return Correlation Matrix",
) -> go.Figure:
    """Full Pearson correlation heatmap of the universe returns."""
    corr = returns.corr()
    tickers = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=tickers, y=tickers,
        text=np.round(z, 2).astype(str),
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
        showscale=True,
    ))
    size = max(500, len(tickers) * 22)
    _apply_layout(fig, title=title, height=size, width=size)
    return fig


def sensitivity_heatmap(
    results_grid: Dict[str, np.ndarray],
    param1_values: List[float],
    param2_values: List[int],
    param1_label: str = "entry_z",
    param2_label: str = "max_half_life",
) -> go.Figure:
    """Three-panel heatmap: Sharpe / CAGR / Max DD over the parameter grid."""
    metrics = [
        ("sharpe", "Sharpe Ratio",    "RdYlGn"),
        ("cagr",   "CAGR (%)",        "RdYlGn"),
        ("max_dd", "Max Drawdown (%)", "RdYlGn_r"),
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Sharpe Ratio", "CAGR (%)", "Max Drawdown (%)"],
        horizontal_spacing=0.08,
    )

    x_labels = [str(v) for v in param2_values]
    y_labels = [str(v) for v in param1_values]

    for col_idx, (key, label, colorscale) in enumerate(metrics, start=1):
        z = results_grid.get(key, np.zeros((len(param1_values), len(param2_values))))
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labels, y=y_labels,
                text=np.round(z, 2).astype(str),
                texttemplate="%{text}",
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=label, x=0.25 + (col_idx - 1) * 0.38, len=0.8),
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text=param2_label, row=1, col=col_idx)
        fig.update_yaxes(title_text=param1_label if col_idx == 1 else "", row=1, col=col_idx)

    fig.update_layout(
        template=_TEMPLATE,
        title="Sensitivity: entry_z × max_half_life",
        height=420,
    )
    return fig


def walk_forward_chart(
    oos_equity: pd.Series,
    fold_summary: pd.DataFrame,
) -> go.Figure:
    """OOS equity curve with fold boundary lines + per-fold Sharpe bar chart below."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.65, 0.35],
        subplot_titles=["OOS Equity Curve", "Per-Fold Sharpe Ratio"],
    )

    eq = oos_equity * 100.0
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name="OOS Equity",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.12)",
    ), row=1, col=1)

    # vertical lines at fold boundaries
    if not fold_summary.empty and "test_start" in fold_summary.columns:
        for _, row in fold_summary.iterrows():
            fig.add_vline(
                x=str(row["test_start"])[:10],
                line=dict(color=_GREY, width=1, dash="dot"),
                row=1, col=1,
            )

    # per-fold sharpe bars — green if positive, red if not
    if not fold_summary.empty:
        sharpe_vals = fold_summary.get("oos_sharpe", pd.Series()).tolist()
        colours = [_GREEN if v >= 0 else _RED for v in sharpe_vals]
        fig.add_trace(go.Bar(
            x=[str(r["test_start"]) for _, r in fold_summary.iterrows()],
            y=sharpe_vals,
            name="Fold Sharpe",
            marker=dict(color=colours),
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="white", width=0.5), row=2, col=1)

    fig.update_layout(template=_TEMPLATE, height=550, showlegend=False)
    return fig


def _apply_layout(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "Date",
    yaxis_title: str = "",
    height: int = 450,
    width: Optional[int] = None,
) -> None:
    """Standard dark layout applied to every chart."""
    kwargs = dict(
        template=_TEMPLATE,
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title=xaxis_title, showgrid=False),
        yaxis=dict(title=yaxis_title, showgrid=True, gridcolor="rgba(80,80,80,0.4)"),
        height=height,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if width is not None:
        kwargs["width"] = width
    fig.update_layout(**kwargs)


def _add_regime_shading(
    fig: go.Figure,
    dates: pd.DatetimeIndex,
    regime_vals: np.ndarray,
    row: int,
) -> None:
    """Shade grey rectangles behind each risk-off (regime=0) period."""
    n = len(dates)
    in_risk_off = False
    start_idx = 0

    for i in range(n):
        if regime_vals[i] == 0 and not in_risk_off:
            in_risk_off = True
            start_idx = i
        elif regime_vals[i] == 1 and in_risk_off:
            in_risk_off = False
            fig.add_vrect(
                x0=str(dates[start_idx])[:10], x1=str(dates[i])[:10],
                fillcolor=_GREY, layer="below", line_width=0,
                row=row, col=1,
            )

    if in_risk_off:
        fig.add_vrect(
            x0=str(dates[start_idx])[:10], x1=str(dates[-1])[:10],
            fillcolor=_GREY, layer="below", line_width=0,
            row=row, col=1,
        )


def _add_trade_markers(
    fig: go.Figure,
    trades: list,
    row: int,
) -> None:
    """Up triangles at entries, down triangles at exits on the z-score panel."""
    if not trades:
        return
    # convert to string to avoid kaleido JSON serialization issues with pd.Timestamp
    entry_dates = [str(t.entry_date)[:10] for t in trades]
    exit_dates = [str(t.exit_date)[:10] for t in trades]

    fig.add_trace(go.Scatter(
        x=entry_dates, y=[0.0] * len(entry_dates),
        mode="markers", name="Entry",
        marker=dict(symbol="triangle-up", size=8, color=_GREEN),
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=exit_dates, y=[0.0] * len(exit_dates),
        mode="markers", name="Exit",
        marker=dict(symbol="triangle-down", size=8, color=_RED),
    ), row=row, col=1)


def _empty_figure(message: str) -> go.Figure:
    """Placeholder figure with a text annotation — used when there's no data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="grey"),
    )
    fig.update_layout(template=_TEMPLATE, height=300)
    return fig
