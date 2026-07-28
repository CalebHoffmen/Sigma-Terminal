import pandas as pd


def calculate_portfolio_history(
    historical_prices: pd.DataFrame,
    holdings: pd.DataFrame,
    benchmark: str,
):
    portfolio_values = pd.DataFrame(index=historical_prices.index)

    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]

        if ticker in historical_prices.columns:
            portfolio_values[ticker] = historical_prices[ticker] * shares

    portfolio_series = portfolio_values.sum(axis=1)

    benchmark_series = historical_prices[benchmark]

    portfolio_index = portfolio_series / portfolio_series.iloc[0] * 100
    benchmark_index = benchmark_series / benchmark_series.iloc[0] * 100

    return portfolio_index, benchmark_index