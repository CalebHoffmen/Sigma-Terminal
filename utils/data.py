from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=300)
def download_current_prices(
    tickers: tuple[str, ...],
) -> pd.Series:
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
                {
                    tickers[0]: float(
                        close_data.dropna().iloc[-1]
                    )
                }
            )

        latest_prices = {}

        for ticker in tickers:
            if ticker in close_data.columns:
                valid_prices = close_data[ticker].dropna()

                if not valid_prices.empty:
                    latest_prices[ticker] = float(
                        valid_prices.iloc[-1]
                    )

        return pd.Series(latest_prices, dtype=float)

    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=900)
def download_historical_prices(
    tickers: tuple[str, ...],
    benchmark: str,
    period: str,
) -> pd.DataFrame:
    symbols = list(
        dict.fromkeys([*tickers, benchmark])
    )

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
            close_data = close_data.to_frame(
                symbols[0]
            )

        return close_data.ffill().dropna(how="all")

    except Exception:
        return pd.DataFrame()