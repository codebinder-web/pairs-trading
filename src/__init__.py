"""
src — all the heavy lifting for the pairs trading system.

data_loader          : price download, parquet caching, log returns
factor_neutralisation: strip market + sector beta via OLS before cointegration testing
pair_selection       : 5-stage cointegration screening pipeline
ou_process           : OU parameter fitting (OLS/MLE) and optimal thresholds
kalman_filter        : online Kalman filter for time-varying hedge ratio
regime_detector      : VIX threshold + HMM for market regime classification
strategy             : stateful FSM signal generator with regime gating
backtest             : vectorised single-pair and portfolio backtester
analytics            : full tearsheet of performance metrics, built from scratch
walk_forward         : rolling OOS validation with strict no-lookahead
visualisation        : Plotly chart builders for the Streamlit dashboard
sensitivity_analysis : 2D grid sweep over entry_z × max_half_life
"""
