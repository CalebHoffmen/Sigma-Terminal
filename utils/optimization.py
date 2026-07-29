import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    sharpe_ratio = (
        (portfolio_return - risk_free_rate) / portfolio_volatility
        if portfolio_volatility != 0
        else 0
    )

    return portfolio_return, portfolio_volatility, sharpe_ratio


def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        _, _, sharpe = portfolio_performance(
            weights,
            mean_returns,
            cov_matrix,
            risk_free_rate,
        )
        return -sharpe

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def minimum_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1,
    }

    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x
import numpy as np
import pandas as pd


def simulate_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
    num_portfolios: int = 5000,
):
    results = []

    num_assets = len(mean_returns)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)

        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        sharpe_ratio = (
            (portfolio_return - risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )

        results.append({
            "Return": portfolio_return,
            "Volatility": portfolio_volatility,
            "Sharpe": sharpe_ratio,
            "Weights": weights,
        })

    return pd.DataFrame(results)