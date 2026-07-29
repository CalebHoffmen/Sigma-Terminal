from utils.optimization import (
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_performance,
)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from utils.optimization import (
    max_sharpe_portfolio,
    minimum_variance_portfolio,
    portfolio_performance,
    simulate_portfolios,
)

st.title("Portfolio Optimization")

tickers_input = st.text_input(
    "Enter tickers separated by commas",
    "AAPL,MSFT,NVDA,GOOGL"
)

tickers = [
    ticker.strip().upper()
    for ticker in tickers_input.split(",")
    if ticker.strip()
]

years = st.slider(
    "Historical lookback (years)",
    min_value=1,
    max_value=10,
    value=3,
)

risk_free_rate = st.number_input(
    "Risk-free rate",
    min_value=0.0,
    max_value=0.20,
    value=0.04,
    step=0.005,
    format="%.3f",
)

if st.button("Optimize Portfolio"):

    if len(tickers) < 2:
        st.error("Enter at least two ticker symbols.")
        st.stop()

    prices = yf.download(
        tickers,
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna()

    returns = prices.pct_change().dropna()

    # Annualized expected returns and covariance matrix
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    simulated = simulate_portfolios(
        mean_returns,
        cov_matrix,
        risk_free_rate,
        num_portfolios=5000,
   )

    max_sharpe_weights = max_sharpe_portfolio(
        mean_returns,
        cov_matrix,
        risk_free_rate,
    )

    min_variance_weights = minimum_variance_portfolio(
        mean_returns,
        cov_matrix,
    )

    max_return, max_vol, max_sharpe = portfolio_performance(
        max_sharpe_weights,
        mean_returns,
        cov_matrix,
        risk_free_rate,
    )

    min_return, min_vol, min_sharpe = portfolio_performance(
        min_variance_weights,
        mean_returns,
        cov_matrix,
        risk_free_rate,
    )
    fig = px.scatter(
        simulated,
        x="Volatility",
        y="Return",
        color="Sharpe",
        title="Efficient Frontier",
        labels={
            "Volatility": "Annualized Volatility",
            "Return": "Expected Annual Return",
            "Sharpe": "Sharpe Ratio",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=[max_vol],
            y=[max_return],
            mode="markers",
            marker=dict(size=14, symbol="star"),
            name="Maximum Sharpe",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_vol],
            y=[min_return],
            mode="markers",
            marker=dict(size=14, symbol="diamond"),
            name="Minimum Variance",
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Maximum Sharpe Portfolio")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Expected Return",
        f"{max_return:.2%}"
    )

    col2.metric(
        "Volatility",
        f"{max_vol:.2%}"
    )

    col3.metric(
        "Sharpe Ratio",
        f"{max_sharpe:.2f}"
    )

    max_weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": max_sharpe_weights,
    })

    max_weights_df["Weight"] = max_weights_df["Weight"].map(
        lambda x: f"{x:.2%}"
    )

    st.dataframe(
        max_weights_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Minimum Variance Portfolio")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Expected Return",
        f"{min_return:.2%}"
    )

    col2.metric(
        "Volatility",
        f"{min_vol:.2%}"
    )

    col3.metric(
        "Sharpe Ratio",
        f"{min_sharpe:.2f}"
    )

    min_weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": min_variance_weights,
    })

    min_weights_df["Weight"] = min_weights_df["Weight"].map(
        lambda x: f"{x:.2%}"
    )

    st.dataframe(
        min_weights_df,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------
    # Optimized Portfolio Weights
    # -----------------------------

    st.subheader("Optimized Portfolio Weights")

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Maximum Sharpe": max_sharpe_weights,
        "Minimum Variance": min_variance_weights,
    })

    weights_long = weights_df.melt(
        id_vars="Ticker",
        var_name="Portfolio",
        value_name="Weight",
    )

    fig_weights = px.bar(
        weights_long,
        x="Ticker",
        y="Weight",
        color="Portfolio",
        barmode="group",
        title="Optimal Asset Allocation",
        labels={
            "Weight": "Portfolio Weight",
            "Ticker": "Asset",
        },
    )

    fig_weights.update_yaxes(tickformat=".0%")

    st.plotly_chart(
        fig_weights,
        use_container_width=True,
    )
    st.subheader("Portfolio Comparison")

    equal_weights = np.array(
        [1 / len(tickers)] * len(tickers)
    )

    equal_return, equal_vol, equal_sharpe = portfolio_performance(
        equal_weights,
        mean_returns,
        cov_matrix,
        risk_free_rate,
    )

    comparison_df = pd.DataFrame({
        "Portfolio": [
            "Equal Weight",
            "Maximum Sharpe",
            "Minimum Variance",
        ],
        "Expected Return": [
            equal_return,
            max_return,
            min_return,
        ],
        "Volatility": [
            equal_vol,
            max_vol,
            min_vol,
        ],
        "Sharpe Ratio": [
            equal_sharpe,
            max_sharpe,
            min_sharpe,
        ],
    })

    display_df = comparison_df.copy()

    display_df["Expected Return"] = display_df["Expected Return"].map(
        lambda x: f"{x:.2%}"
    ) 

    display_df["Volatility"] = display_df["Volatility"].map(
        lambda x: f"{x:.2%}"
    )

    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].map(
        lambda x: f"{x:.2f}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    fig_compare = px.scatter(
        comparison_df,
        x="Volatility",
        y="Expected Return",
        text="Portfolio",
        title="Risk vs. Expected Return",
        labels={
            "Volatility": "Annualized Volatility",
            "Expected Return": "Expected Annual Return",
        },
    )

    fig_compare.update_traces(
        marker=dict(size=14),
        textposition="top center",
    )

    fig_compare.update_xaxes(tickformat=".1%")
    fig_compare.update_yaxes(tickformat=".1%")

    st.plotly_chart(
        fig_compare,
        use_container_width=True,
    ) 