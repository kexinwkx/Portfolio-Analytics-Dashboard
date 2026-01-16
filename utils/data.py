import pandas as pd
import yfinance as yf
from typing import List


def fetch_prices(
    tickers: List[str],
    start: str,
    end: str
) -> pd.DataFrame:
    # clean ticker list
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()

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

    # MultiIndex columns: ('Close','AAPL'), ('Adj Close','MSFT'), ...
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
        elif "Adj Close" in df.columns.get_level_values(0):
            close = df["Adj Close"]
        else:
            return pd.DataFrame()
    else:
        # single ticker fallback
        if "Close" in df.columns:
            close = df[["Close"]]
            close.columns = [tickers[0]]
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]]
            close.columns = [tickers[0]]
        else:
            return pd.DataFrame()

    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index)
    return close


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    return prices.pct_change().dropna(how="all")


def equal_weight_portfolio_returns(
    returns_df: pd.DataFrame
) -> pd.Series:
    if returns_df is None or returns_df.empty:
        return pd.Series(dtype=float)

    r = returns_df.dropna(axis=1, how="all")
    if r.empty:
        return pd.Series(dtype=float)

    port = r.mean(axis=1, skipna=True)
    port.name = "Portfolio"
    return port
