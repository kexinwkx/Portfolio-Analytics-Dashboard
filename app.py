import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Analytics Dashboard")
st.caption("Python + Streamlit | Portfolio performance & risk monitoring")

st.markdown(
    """
This dashboard summarizes **portfolio performance and risk** to support basic investment analysis and monitoring.

**Workflow (Session 1–5):** Data → Cleaning → Returns → Portfolio → Risk → Visualization → App

**Portfolio setup:** Equal-weight portfolio across selected tickers (simple and interview-safe).
"""
)

# ----------------------------
# Sidebar: Controls
# ----------------------------
st.sidebar.header("Inputs")

default_tickers = "AAPL,MSFT,JPM,GS"
tickers_text = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)

colA, colB = st.sidebar.columns(2)
start_date = colA.date_input("Start", value=date(2020, 1, 1))
end_date = colB.date_input("End", value=date.today())

if start_date >= end_date:
    st.sidebar.error("Start date must be earlier than End date.")

risk_free = st.sidebar.number_input(
    "Risk-free rate (annual, %)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.25
)
rf_annual = risk_free / 100.0

st.sidebar.divider()
st.sidebar.write("Tip: keep tickers 3–8 for stable output.")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_close_prices(tickers, start, end) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        elif "Adj Close" in df.columns.get_level_values(0):
            close = df["Adj Close"]
        else:
            return pd.DataFrame()
    else:
        # Single ticker
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            close.columns = [tickers[0]]
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]].copy()
            close.columns = [tickers[0]]
        else:
            return pd.DataFrame()

    close = close.sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Session 2: transform price → returns
    return prices.pct_change().dropna(how="all")


def equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.Series:
    # Session 3: equal-weight portfolio
    r = returns_df.dropna(axis=1, how="all")
    if r.empty:
        return pd.Series(dtype=float)

    port = r.mean(axis=1, skipna=True)
    port.name = "Portfolio"
    return port


def equity_curve(returns: pd.Series) -> pd.Series:
    # Session 4: cumulative performance
    eq = (1 + returns.fillna(0)).cumprod()
    eq.name = "Equity"
    return eq


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return float(returns.std(ddof=0) * np.sqrt(trading_days))


def annualized_return(equity: pd.Series) -> float:
    # based on calendar days between first and last
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0:
        return float("nan")

    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (365.0 / n_days) - 1.0)


def sharpe_ratio(returns: pd.Series, rf_annual: float, trading_days: int = 252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = returns - rf_daily

    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")

    return float(excess.mean() / vol * np.sqrt(trading_days))


def plot_equity(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity Curve"
        )
    )
    fig.update_layout(
        title="Portfolio Equity Curve (Cumulative Return)",
        xaxis_title="Date",
        yaxis_title="Growth of $1"
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    peak = equity.cummax()
    dd = equity / peak - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            name="Drawdown"
        )
    )
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown"
    )
    return fig

# ----------------------------
# Run
# ----------------------------
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

if len(tickers) == 0:
    st.warning("Please input at least 1 ticker.")
    st.stop()

with st.spinner("Loading market data..."):
    prices = fetch_close_prices(tickers, start_date, end_date)

if prices.empty:
    st.error("No data returned. Check tickers or date range.")
    st.stop()

# Session 2: Cleaning (basic)
prices = prices.ffill().dropna(how="all")

returns_df = compute_returns(prices)
port_ret = equal_weight_portfolio(returns_df)

if port_ret.empty:
    st.error("Portfolio returns are empty. Try a different date range or tickers.")
    st.stop()

eq = equity_curve(port_ret)

# Metrics (Session 4)
ann_ret = annualized_return(eq)
ann_vol = annualized_vol(port_ret)
mdd = max_drawdown(eq)
sr = sharpe_ratio(port_ret, rf_annual=rf_annual)

# ----------------------------
# Layout
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annualized Return", f"{ann_ret * 100:.2f}%")
c2.metric("Annualized Volatility", f"{ann_vol * 100:.2f}%")
c3.metric("Max Drawdown", f"{mdd * 100:.2f}%")
c4.metric("Sharpe (rf adj.)", f"{sr:.2f}")

left, right = st.columns([2, 1])

with left:
    st.plotly_chart(plot_equity(eq), use_container_width=True)
    st.plotly_chart(plot_drawdown(eq), use_container_width=True)

with right:
    st.subheader("What you’re looking at")
    st.markdown(
        """
- **Return**: equity curve shows cumulative growth of $1 invested.
- **Risk**: volatility captures overall variability; drawdown shows worst peak-to-trough loss.
- **Trade-off**: higher return often comes with higher volatility / deeper drawdowns.
"""
    )

    st.subheader("Data Preview")
    st.write("Close prices (head):")
    st.dataframe(prices.head())

    st.write("Daily returns (head):")
    st.dataframe(returns_df.head())

    st.download_button(
        "Download portfolio returns (CSV)",
        data=port_ret.to_frame().to_csv(index=True).encode("utf-8"),
        file_name="portfolio_returns.csv",
        mime="text/csv",
    )
