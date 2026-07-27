import pandas as pd
import yfinance as yf

from utils.analytics import (
    annualized_return,
    annualized_volatility,
    cumulative_returns,
    daily_returns,
    sharpe_ratio,
)

def test_with_known_prices() -> None:
    """
    Test daily_returns using prices where the expected returns are known.
    """
    prices = pd.Series([100.0, 110.0, 99.0])

    results = daily_returns(prices)

    expected_first_return = 0.10
    expected_second_return = -0.10

    assert abs(results.iloc[0] - expected_first_return) < 0.000001
    assert abs(results.iloc[1] - expected_second_return) < 0.000001

    print("Known-price test passed.")


def main() -> None:
    test_with_known_prices()

    ticker = "AAPL"

    print(f"\nDownloading price data for {ticker}...")

    data = yf.download(
        ticker,
        period="1y",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        print("No market data was downloaded.")
        return

    prices = data["Close"]

    # Some yfinance versions return a one-column DataFrame.
    if hasattr(prices, "columns"):
        prices = prices.squeeze("columns")

    returns = daily_returns(prices)
    cum_returns = cumulative_returns(prices)
    volatility = annualized_volatility(prices)
    print(f"\nAnnualized volatility: {volatility:.2%}")

    print("\nClosing prices:")
    print(prices.head())

    print("\nDaily returns:")
    print(returns.head())
    print("\nCumulative returns:")
    print((cum_returns.head() * 100).round(2).astype(str) + "%")

    print("\nDaily returns as percentages:")
    print((returns.head() * 100).round(2).astype(str) + "%")

    print("\nReal-market-data test passed.")

    annual_return = annualized_return(prices)
    print(f"Annualized return: {annual_return:.2%}")
    print(f"Annualized volatility: {volatility:.2%}")

    sharpe = sharpe_ratio(prices)
    print(f"Annualized return: {annual_return:.2%}")
    print(f"Annualized volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")


if __name__ == "__main__":
    main()