import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(
    returns: pd.Series,
    start_value: float = 1.0
) -> pd.Series:
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    eq = (1 + returns.fillna(0)).cumprod() * start_value
    eq.name = "Equity"
    return eq


def annualized_return(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return float("nan")

    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0:
        return float("nan")

    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (365.0 / n_days) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return float("nan")

    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return float("nan")

    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0
) -> float:
    if returns is None or returns.empty:
        return float("nan")

    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = returns - rf_daily
    vol = excess.std(ddof=0)

    if vol == 0 or np.isnan(vol):
        return float("nan")

    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))
