from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Portfolio | OptiQuant Terminal",
    page_icon="💼",
    layout="wide",
)


DEFAULT_HOLDINGS = pd.DataFrame(
    {
        "Ticker": ["SPY", "QQQ", "TLT", "GLD"],
        "Shares": [10.0, 5.0, 15.0, 8.0],
        "Average Cost": [500.0, 450.0, 95.0, 220.0],
    }
)


def clean_holdings(raw_holdings: pd.DataFrame) -> pd.DataFrame:
    holdings = raw_holdings.copy()

    required_columns = ["Ticker", "Shares", "Average Cost"]

    for column in required_columns:
        if column not in holdings.columns:
            holdings[column] = np.nan

    holdings["Ticker"] = (
        holdings["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    holdings["Shares"] = pd.to_numeric(
        holdings["Shares"],
        errors="coerce",
    )

    holdings["Average Cost"] = pd.to_numeric(
        holdings["Average Cost"],
        errors="coerce",
    )

    holdings = holdings.dropna(
        subset=["Ticker", "Shares", "Average Cost"]
    )

    holdings = holdings[
        (holdings["Ticker"] != "")
        & (holdings["Ticker"] != "NAN")
        & (holdings["Shares"] > 0)
        & (holdings["Average Cost"] >= 0)
    ]

    return holdings.reset_index(drop=True)


@st.cache_data(ttl=300)
def download_current_prices(tickers: tuple[str, ...]) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)

    try:
        data = yf.download(
            list(tickers),
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )

        if data.empty:
            return pd.Series(dtype=float)

        close_data = data["Close"]

        if isinstance(close_data, pd.Series):
            return pd.Series(
                {tickers[0]: float(close_data.dropna().iloc[-1])}
            )

        latest_prices = {}

        for ticker in tickers:
            if ticker in close_data.columns:
                valid_prices = close_data[ticker].dropna()

                if not valid_prices.empty:
                    latest_prices[ticker] = float(valid_prices.iloc[-1])

        return pd.Series(latest_prices, dtype=float)

    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=900)
def download_historical_prices(
    tickers: tuple[str, ...],
    benchmark: str,
    period: str,
) -> pd.DataFrame:
    symbols = list(dict.fromkeys([*tickers, benchmark]))

    if not symbols:
        return pd.DataFrame()

    try:
        data = yf.download(
            symbols,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame()

        close_data = data["Close"]

        if isinstance(close_data, pd.Series):
            close_data = close_data.to_frame(symbols[0])

        return close_data.ffill().dropna(how="all")

    except Exception:
        return pd.DataFrame()


def calculate_portfolio_history(
    prices: pd.DataFrame,
    holdings: pd.DataFrame,
    benchmark: str,
) -> tuple[pd.Series, pd.Series]:
    portfolio_tickers = [
        ticker
        for ticker in holdings["Ticker"]
        if ticker in prices.columns
    ]

    if not portfolio_tickers:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    shares = (
        holdings.set_index("Ticker")
        .loc[portfolio_tickers, "Shares"]
    )

    portfolio_value = (
        prices[portfolio_tickers]
        .mul(shares, axis=1)
        .sum(axis=1)
    )

    portfolio_return_index = (
        portfolio_value / portfolio_value.iloc[0]
    )

    if benchmark in prices.columns:
        benchmark_prices = prices[benchmark].dropna()
        benchmark_return_index = (
            benchmark_prices / benchmark_prices.iloc[0]
        )
    else:
        benchmark_return_index = pd.Series(dtype=float)

    return portfolio_return_index, benchmark_return_index


def calculate_risk_metrics(
    return_index: pd.Series,
) -> dict[str, float]:
    if return_index.empty or len(return_index) < 2:
        return {
            "Total Return": np.nan,
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Maximum Drawdown": np.nan,
        }

    daily_returns = return_index.pct_change().dropna()

    total_return = return_index.iloc[-1] - 1

    elapsed_years = max(len(return_index) / 252, 1 / 252)

    annualized_return = (
        return_index.iloc[-1] ** (1 / elapsed_years)
    ) - 1

    annualized_volatility = (
        daily_returns.std() * np.sqrt(252)
    )

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    running_peak = return_index.cummax()
    drawdown = return_index / running_peak - 1
    maximum_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
    }


st.title("💼 Portfolio Tracker")

st.write(
    "Enter your holdings below to calculate market value, "
    "profit and loss, allocation, performance, and risk."
)

with st.sidebar:
    st.header("Portfolio Settings")

    benchmark = st.text_input(
        "Benchmark",
        value="SPY",
    ).upper().strip()

    analysis_period = st.selectbox(
        "Performance Period",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )


edited_holdings = st.data_editor(
    DEFAULT_HOLDINGS,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(
            "Ticker",
            help="Yahoo Finance ticker symbol",
            required=True,
        ),
        "Shares": st.column_config.NumberColumn(
            "Shares",
            min_value=0.0,
            step=1.0,
            format="%.4f",
            required=True,
        ),
        "Average Cost": st.column_config.NumberColumn(
            "Average Cost",
            min_value=0.0,
            step=1.0,
            format="$%.2f",
            required=True,
        ),
    },
    hide_index=True,
)

holdings = clean_holdings(edited_holdings)

if holdings.empty:
    st.warning("Add at least one valid holding to continue.")
    st.stop()


ticker_tuple = tuple(holdings["Ticker"].unique())

latest_prices = download_current_prices(ticker_tuple)

holdings["Current Price"] = holdings["Ticker"].map(
    latest_prices
)

missing_tickers = holdings.loc[
    holdings["Current Price"].isna(),
    "Ticker",
].tolist()

if missing_tickers:
    st.warning(
        "Current prices could not be loaded for: "
        + ", ".join(missing_tickers)
    )

holdings = holdings.dropna(subset=["Current Price"])

if holdings.empty:
    st.error("No valid market prices were available.")
    st.stop()


holdings["Cost Basis"] = (
    holdings["Shares"] * holdings["Average Cost"]
)

holdings["Market Value"] = (
    holdings["Shares"] * holdings["Current Price"]
)

holdings["Unrealized P&L"] = (
    holdings["Market Value"] - holdings["Cost Basis"]
)

holdings["Return"] = np.where(
    holdings["Cost Basis"] > 0,
    holdings["Unrealized P&L"] / holdings["Cost Basis"],
    np.nan,
)

total_market_value = holdings["Market Value"].sum()
total_cost_basis = holdings["Cost Basis"].sum()
total_profit_loss = holdings["Unrealized P&L"].sum()

portfolio_return = (
    total_profit_loss / total_cost_basis
    if total_cost_basis > 0
    else np.nan
)

holdings["Weight"] = (
    holdings["Market Value"] / total_market_value
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Portfolio Value",
    f"${total_market_value:,.2f}",
)

metric_2.metric(
    "Total Cost Basis",
    f"${total_cost_basis:,.2f}",
)

metric_3.metric(
    "Unrealized P&L",
    f"${total_profit_loss:,.2f}",
)

metric_4.metric(
    "Portfolio Return",
    f"{portfolio_return:.2%}"
    if pd.notna(portfolio_return)
    else "Not available",
)


st.subheader("Holdings Analysis")

formatted_holdings = holdings.copy()

formatted_holdings["Shares"] = formatted_holdings[
    "Shares"
].map(lambda value: f"{value:,.4f}")

for column in [
    "Average Cost",
    "Current Price",
    "Cost Basis",
    "Market Value",
    "Unrealized P&L",
]:
    formatted_holdings[column] = formatted_holdings[
        column
    ].map(lambda value: f"${value:,.2f}")

formatted_holdings["Return"] = formatted_holdings[
    "Return"
].map(lambda value: f"{value:.2%}")

formatted_holdings["Weight"] = formatted_holdings[
    "Weight"
].map(lambda value: f"{value:.2%}")

st.dataframe(
    formatted_holdings,
    use_container_width=True,
    hide_index=True,
)


allocation_tab, profit_loss_tab = st.tabs(
    ["Portfolio Allocation", "Profit and Loss"]
)

with allocation_tab:
    allocation_chart = px.pie(
        holdings,
        names="Ticker",
        values="Market Value",
        title="Portfolio Allocation by Market Value",
        hole=0.45,
    )

    allocation_chart.update_traces(
        textposition="inside",
        textinfo="label+percent",
    )

    st.plotly_chart(
        allocation_chart,
        use_container_width=True,
    )


with profit_loss_tab:
    profit_loss_chart = px.bar(
        holdings,
        x="Ticker",
        y="Unrealized P&L",
        title="Unrealized Profit and Loss by Holding",
        text_auto=".2s",
    )

    profit_loss_chart.update_layout(
        yaxis_title="Unrealized P&L ($)",
    )

    st.plotly_chart(
        profit_loss_chart,
        use_container_width=True,
    )


historical_prices = download_historical_prices(
    ticker_tuple,
    benchmark,
    analysis_period,
)

if historical_prices.empty:
    st.warning(
        "Historical portfolio performance could not be loaded."
    )
else:
    portfolio_index, benchmark_index = (
        calculate_portfolio_history(
            historical_prices,
            holdings,
            benchmark,
        )
    )

    if not portfolio_index.empty:
        st.subheader("Historical Performance")

        performance_chart = go.Figure()

        performance_chart.add_trace(
            go.Scatter(
                x=portfolio_index.index,
                y=portfolio_index,
                mode="lines",
                name="Portfolio",
            )
        )

        if not benchmark_index.empty:
            performance_chart.add_trace(
                go.Scatter(
                    x=benchmark_index.index,
                    y=benchmark_index,
                    mode="lines",
                    name=benchmark,
                )
            )

        performance_chart.update_layout(
            title=(
                f"Growth of $1: Portfolio vs. {benchmark}"
            ),
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            height=500,
            hovermode="x unified",
        )

        st.plotly_chart(
            performance_chart,
            use_container_width=True,
        )

        portfolio_metrics = calculate_risk_metrics(
            portfolio_index
        )

        st.subheader("Portfolio Risk and Performance")

        risk_1, risk_2, risk_3, risk_4, risk_5 = (
            st.columns(5)
        )

        risk_1.metric(
            "Total Return",
            f"{portfolio_metrics['Total Return']:.2%}",
        )

        risk_2.metric(
            "Annualized Return",
            f"{portfolio_metrics['Annualized Return']:.2%}",
        )

        risk_3.metric(
            "Annualized Volatility",
            f"{portfolio_metrics['Annualized Volatility']:.2%}",
        )

        risk_4.metric(
            "Sharpe Ratio",
            f"{portfolio_metrics['Sharpe Ratio']:.2f}",
        )

        risk_5.metric(
            "Maximum Drawdown",
            f"{portfolio_metrics['Maximum Drawdown']:.2%}",
        )


csv_data = holdings.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Portfolio Analysis as CSV",
    data=csv_data,
    file_name="optiquant_portfolio_analysis.csv",
    mime="text/csv",
)

st.caption(
    "Prices are obtained from Yahoo Finance and may be "
    "delayed, adjusted, incomplete, or unavailable."
)