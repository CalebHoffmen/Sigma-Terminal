import pandas as pd
import numpy as np


class risk_free_rate:
    """
    Lightweight wrapper for a constant annual risk-free rate.

    This class provides both the annualized rate and the equivalent
    per-period rate for use in Sharpe/Sortino calculations.
    """

    def __init__(self, annual_rate: float = 0.04) -> None:
        if annual_rate < 0:
            raise ValueError("The annual risk-free rate must be non-negative.")

        self.annual_rate = float(annual_rate)

    def get_rate(self, periods_per_year: int = 252) -> float:
        """
        Return the annualized risk-free rate.

        Parameters
        ----------
        periods_per_year : int
            Number of periods per year. Included for API compatibility.

        Returns
        -------
        float
            The annual risk-free rate as a decimal.
        """
        return self.annual_rate

    def get_period_rate(self, periods_per_year: int = 252) -> float:
        """
        Return the equivalent per-period risk-free rate.

        Parameters
        ----------
        periods_per_year : int
            Number of periods per year for conversion.

        Returns
        -------
        float
            Risk-free rate per period, expressed as a decimal.
        """
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer.")

        return self.annual_rate / periods_per_year


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


def cagr(return_index):
    if return_index.empty or len(return_index) < 2:
        return float("nan")

    start_value = return_index.iloc[0]
    end_value = return_index.iloc[-1]

    days = (return_index.index[-1] - return_index.index[0]).days

    if days <= 0 or start_value <= 0:
        return float("nan")

    years = days / 365.25

    return (end_value / start_value) ** (1 / years) - 1


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> float:
    downside_returns = returns[returns < 0]

    if downside_returns.empty:
        return float("nan")

    downside_deviation = downside_returns.std() * np.sqrt(252)
    annualized_return = returns.mean() * 252

    if downside_deviation == 0:
        return float("nan")

    return (
        annualized_return - risk_free_rate
    ) / downside_deviation


def alpha_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> tuple[float, float]:
    """
    Calculate annualized portfolio alpha and beta
    relative to a benchmark.
    """

    combined = pd.concat(
        [portfolio_returns, benchmark_returns],
        axis=1,
    ).dropna()

    if len(combined) < 2:
        return float("nan"), float("nan")

    portfolio = combined.iloc[:, 0]
    benchmark = combined.iloc[:, 1]

    benchmark_variance = benchmark.var()

    if benchmark_variance == 0:
        return float("nan"), float("nan")

    beta = portfolio.cov(benchmark) / benchmark_variance

    portfolio_annual_return = portfolio.mean() * 252
    benchmark_annual_return = benchmark.mean() * 252

    alpha = portfolio_annual_return - (
        risk_free_rate
        + beta * (benchmark_annual_return - risk_free_rate)
    )

    return alpha, beta