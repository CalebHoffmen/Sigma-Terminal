import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate the daily percentage returns of an asset.

    Parameters
    ----------
    prices : pd.Series
        Historical closing prices.

    Returns
    -------
    pd.Series
        Daily percentage returns with missing values removed.
    """
    if prices.empty:
        raise ValueError("The price series is empty.")

    returns = prices.pct_change()

    return returns.dropna()
def cumulative_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns over time.

    Parameters
    ----------
    prices : pd.Series
        Historical closing prices.

    Returns
    -------
    pd.Series
        Cumulative return series.
    """
    returns = daily_returns(prices)

    return (1 + returns).cumprod() - 1

def annualized_volatility(
    prices: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Calculate annualized volatility from daily returns.

    Parameters
    ----------
    prices : pd.Series
        Historical closing prices.

    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualized volatility as a decimal.
    """
    returns = daily_returns(prices)

    return returns.std() * (trading_days ** 0.5)
def annualized_return(
    prices: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    Calculate annualized return using geometric compounding.

    Parameters
    ----------
    prices : pd.Series
        Historical closing prices.

    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualized return as a decimal.
    """
    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    total_return = prices.iloc[-1] / prices.iloc[0]

    number_of_periods = len(prices) - 1

    return total_return ** (trading_days / number_of_periods) - 1
def sharpe_ratio(
    prices: pd.Series,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    """
    annual_return = annualized_return(prices)
    annual_vol = annualized_volatility(prices)

    if annual_vol == 0:
        return 0.0

    return (annual_return - risk_free_rate) / annual_vol

import numpy as np
import pandas as pd


def portfolio_risk_metrics(
    return_index: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, float]:
    """
    Calculate portfolio performance and risk metrics.

    Parameters
    ----------
    return_index:
        Growth-of-$1 series, normally beginning near 1.0.

    risk_free_rate:
        Annual risk-free rate expressed as a decimal.
        Example: 0.04 means 4%.

    trading_days:
        Number of trading periods per year.

    Returns
    -------
    dict[str, float]
        Total return, annualized return, annualized volatility,
        Sharpe ratio, and maximum drawdown.
    """
    clean_index = return_index.dropna()

    if len(clean_index) < 2:
        return {
            "Total Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
        }

    returns = clean_index.pct_change().dropna()

    total_return = clean_index.iloc[-1] / clean_index.iloc[0] - 1

    number_of_periods = len(returns)

    annualized_return = (
        (clean_index.iloc[-1] / clean_index.iloc[0])
        ** (trading_days / number_of_periods)
        - 1
    )

    annualized_volatility = (
        returns.std() * np.sqrt(trading_days)
    )

    if annualized_volatility > 0:
        sharpe_ratio = (
            annualized_return - risk_free_rate
        ) / annualized_volatility
    else:
        sharpe_ratio = np.nan

    running_peak = clean_index.cummax()
    drawdown = clean_index / running_peak - 1
    maximum_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
    }

    def calculate_drawdown(portfolio_values):
        running_max = portfolio_values.cummax()
        drawdown = portfolio_values / running_max - 1
        return drawdown

    