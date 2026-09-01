from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from utils.portfolio import calculate_portfolio_history
from utils.analytics import (
    calculate_drawdown,
    portfolio_risk_metrics,
    cagr,
    sortino_ratio,
)
from utils.data import (
    download_current_prices,
    download_historical_prices,
)
from utils.stress_test import stress_test_portfolio
from utils.scenarios import SCENARIOS
st.set_page_config(
    page_title="Portfolio | Sigma Terminal",
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
    drawdown = calculate_drawdown(portfolio_index)
    max_drawdown = drawdown.min()

    st.subheader("Drawdown Analysis")

    st.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2%}"
    )

    fig_drawdown = go.Figure()

    fig_drawdown.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode="lines",
            name="Portfolio Drawdown",
        )
    )

    fig_drawdown.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
    )

    st.plotly_chart(
    fig_drawdown,
    use_container_width=True,
    key="portfolio_drawdown_chart",
)
    st.subheader("Asset Correlation")

    asset_returns = historical_prices.pct_change().dropna()

    correlation_matrix = asset_returns.corr()

    fig_correlation = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Portfolio Correlation Matrix",
    )

    st.plotly_chart(
        fig_correlation,
        use_container_width=True,
        key="portfolio_correlation_heatmap",
    ) 

    stress = st.slider(
        "Market Shock (%)",
        -50,
        0,
        -20,
    )

    stressed = stress_test_portfolio(
        holdings,
        stress / 100,
    )

    st.subheader("Portfolio Stress Test")

    st.dataframe(
        stressed[
            [
                "Ticker",
                "Market Value",
                "Scenario Value",
                "Loss",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Scenario Stress Test")

    scenario_name = st.selectbox(
        "Select Scenario",
        list(SCENARIOS.keys()),
        key="stress_scenario_selector",
    )

    scenario = SCENARIOS[scenario_name]

    scenario_results = holdings.copy()

    scenario_results["Shock"] = scenario_results["Ticker"].apply(
        lambda ticker: scenario.get(
            ticker,
            scenario.get("default", 0.0),
        )
    )

    scenario_results["Scenario Value"] = (
        scenario_results["Market Value"]
        * (1 + scenario_results["Shock"])
    )

    scenario_results["Scenario P&L"] = (
        scenario_results["Scenario Value"]
      - scenario_results["Market Value"]
    )

    scenario_results["Shock"] = scenario_results["Shock"].map(
        lambda x: f"{x:.1%}"
    )

    st.dataframe(
        scenario_results[
            [
                "Ticker",
                "Market Value",
                "Shock",
                "Scenario Value",
                "Scenario P&L",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    # Calculate daily portfolio returns
    portfolio_returns = portfolio_index.pct_change().dropna()
    portfolio_cagr = cagr(portfolio_index)
    portfolio_sortino = sortino_ratio(portfolio_returns)
    # Calculate 30-day rolling annualized volatility
    rolling_vol = (
        portfolio_returns
       .rolling(30)
        .std()
        * np.sqrt(252)
    )

    st.subheader("Rolling Risk")

    fig_volatility = go.Figure()

    fig_volatility.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode="lines",
            name="30-Day Rolling Volatility",
        )
    )

    fig_volatility.update_layout(
        title="30-Day Rolling Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
    )

    fig_volatility.update_yaxes(tickformat=".1%")

    st.plotly_chart(
        fig_volatility,
        use_container_width=True,
        key="rolling_volatility_chart",
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

        portfolio_metrics = portfolio_risk_metrics(
            portfolio_index
        )

        st.subheader("Portfolio Risk and Performance")

        risk_1, risk_2, risk_3, risk_4, risk_5, risk_6, risk_7 = st.columns(7)
        

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
        risk_6.metric(
            "CAGR",
            f"{portfolio_cagr:.2%}"
            if pd.notna(portfolio_cagr)
            else "N/A",
        )

        risk_7.metric(
            "Sortino Ratio",
             f"{portfolio_sortino:.2f}"
             if pd.notna(portfolio_sortino)
             else "N/A",
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