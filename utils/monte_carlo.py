import numpy as np
import pandas as pd


def monte_carlo_simulation(
    returns: pd.Series,
    starting_value: float,
    days: int = 252,
    simulations: int = 1000,
) -> pd.DataFrame:

    clean_returns = returns.dropna()

    if clean_returns.empty:
        return pd.DataFrame()

    mean_return = clean_returns.mean()
    volatility = clean_returns.std()

    random_returns = np.random.normal(
        mean_return,
        volatility,
        size=(days, simulations),
    )

    growth = np.cumprod(
        1 + random_returns,
        axis=0,
    )

    simulated_values = starting_value * growth

    return pd.DataFrame(simulated_values)