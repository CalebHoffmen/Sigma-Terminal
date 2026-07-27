import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Markets | Sigma Terminal",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=900)
def load_market_data(
    symbol: str,
    selected_period: str,
    selected_interval: str,
) -> pd.DataFrame:
    try:
        data = yf.download(
            symbol,
            period=selected_period,
            interval=selected_interval,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data.dropna(how="all")

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def load_company_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info
    except Exception:
        return {}


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    return 100 - (100 / (1 + relative_strength))


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["SMA_20"] = result["Close"].rolling(20).mean()
    result["SMA_50"] = result["Close"].rolling(50).mean()

    result["EMA_12"] = result["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    result["EMA_26"] = result["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    result["MACD"] = result["EMA_12"] - result["EMA_26"]

    result["MACD_SIGNAL"] = result["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    result["MACD_HISTOGRAM"] = (
        result["MACD"] - result["MACD_SIGNAL"]
    )

    result["RSI_14"] = calculate_rsi(result["Close"])

    rolling_mean = result["Close"].rolling(20).mean()
    rolling_std = result["Close"].rolling(20).std()

    result["BB_UPPER"] = rolling_mean + 2 * rolling_std
    result["BB_LOWER"] = rolling_mean - 2 * rolling_std

    result["DAILY_RETURN"] = result["Close"].pct_change()

    result["ROLLING_VOLATILITY_20"] = (
        result["DAILY_RETURN"].rolling(20).std() * np.sqrt(252)
    )

    return result


def format_market_cap(value: float | None) -> str:
    if value is None:
        return "Not available"

    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.0f}"


st.title("📈 Markets Dashboard")

with st.sidebar:
    st.header("Market Controls")

    ticker = st.text_input(
        "Ticker",
        value="AAPL",
    ).upper().strip()

    period = st.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
        index=3,
    )

    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo"],
        index=0,
    )

    show_sma_20 = st.checkbox(
        "20-Period SMA",
        value=True,
    )

    show_sma_50 = st.checkbox(
        "50-Period SMA",
        value=True,
    )

    show_bollinger_bands = st.checkbox(
        "Bollinger Bands",
        value=True,
    )


market_data = load_market_data(
    ticker,
    period,
    interval,
)

company_info = load_company_info(ticker)

if market_data.empty:
    st.error(
        f"No market data was found for {ticker}. "
        "Check the ticker and try again."
    )
    st.stop()


market_data = calculate_indicators(market_data)

latest_close = float(market_data["Close"].iloc[-1])

if len(market_data) > 1:
    previous_close = float(market_data["Close"].iloc[-2])
else:
    previous_close = latest_close

price_change = latest_close - previous_close

percent_change = (
    price_change / previous_close
    if previous_close != 0
    else 0
)

company_name = company_info.get("longName", ticker)
sector = company_info.get("sector", "Not available")
industry = company_info.get("industry", "Not available")
market_cap = company_info.get("marketCap")
trailing_pe = company_info.get("trailingPE")
dividend_yield = company_info.get("dividendYield")
beta = company_info.get("beta")


st.subheader(company_name)

st.caption(
    f"{ticker} | {sector} | {industry}"
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Latest Close",
    f"${latest_close:,.2f}",
    f"{price_change:,.2f} ({percent_change:.2%})",
)

metric_2.metric(
    "Market Cap",
    format_market_cap(market_cap),
)

metric_3.metric(
    "Trailing P/E",
    f"{trailing_pe:.2f}"
    if trailing_pe is not None
    else "Not available",
)

metric_4.metric(
    "Beta",
    f"{beta:.2f}"
    if beta is not None
    else "Not available",
)


metric_5, metric_6, metric_7, metric_8 = st.columns(4)

metric_5.metric(
    "Period High",
    f"${float(market_data['High'].max()):,.2f}",
)

metric_6.metric(
    "Period Low",
    f"${float(market_data['Low'].min()):,.2f}",
)

latest_rsi = market_data["RSI_14"].iloc[-1]

metric_7.metric(
    "RSI (14)",
    f"{latest_rsi:.2f}"
    if pd.notna(latest_rsi)
    else "Not available",
)

metric_8.metric(
    "Dividend Yield",
    f"{dividend_yield:.2%}"
    if dividend_yield is not None
    else "Not available",
)


price_chart = go.Figure()

price_chart.add_trace(
    go.Candlestick(
        x=market_data.index,
        open=market_data["Open"],
        high=market_data["High"],
        low=market_data["Low"],
        close=market_data["Close"],
        name=ticker,
    )
)

if show_sma_20:
    price_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["SMA_20"],
            mode="lines",
            name="SMA 20",
        )
    )

if show_sma_50:
    price_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["SMA_50"],
            mode="lines",
            name="SMA 50",
        )
    )

if show_bollinger_bands:
    price_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["BB_UPPER"],
            mode="lines",
            name="Upper Bollinger Band",
            line=dict(width=1),
        )
    )

    price_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["BB_LOWER"],
            mode="lines",
            name="Lower Bollinger Band",
            line=dict(width=1),
            fill="tonexty",
            fillcolor="rgba(100, 100, 100, 0.08)",
        )
    )

price_chart.update_layout(
    title=f"{company_name} Price History",
    xaxis_title="Date",
    yaxis_title="Price",
    height=650,
    xaxis_rangeslider_visible=False,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    price_chart,
    use_container_width=True,
)


chart_tab_1, chart_tab_2, chart_tab_3, chart_tab_4 = st.tabs(
    [
        "Volume",
        "RSI",
        "MACD",
        "Rolling Volatility",
    ]
)


with chart_tab_1:
    volume_chart = go.Figure()

    volume_chart.add_trace(
        go.Bar(
            x=market_data.index,
            y=market_data["Volume"],
            name="Volume",
        )
    )

    volume_chart.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Volume",
    )

    st.plotly_chart(
        volume_chart,
        use_container_width=True,
    )


with chart_tab_2:
    rsi_chart = go.Figure()

    rsi_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["RSI_14"],
            mode="lines",
            name="RSI 14",
        )
    )

    rsi_chart.add_hline(
        y=70,
        line_dash="dash",
        annotation_text="Overbought",
    )

    rsi_chart.add_hline(
        y=30,
        line_dash="dash",
        annotation_text="Oversold",
    )

    rsi_chart.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        rsi_chart,
        use_container_width=True,
    )


with chart_tab_3:
    macd_chart = go.Figure()

    macd_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["MACD"],
            mode="lines",
            name="MACD",
        )
    )

    macd_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["MACD_SIGNAL"],
            mode="lines",
            name="Signal",
        )
    )

    macd_chart.add_trace(
        go.Bar(
            x=market_data.index,
            y=market_data["MACD_HISTOGRAM"],
            name="Histogram",
        )
    )

    macd_chart.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="MACD",
    )

    st.plotly_chart(
        macd_chart,
        use_container_width=True,
    )


with chart_tab_4:
    volatility_chart = go.Figure()

    volatility_chart.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data["ROLLING_VOLATILITY_20"],
            mode="lines",
            name="20-Period Annualized Volatility",
        )
    )

    volatility_chart.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
    )

    st.plotly_chart(
        volatility_chart,
        use_container_width=True,
    )


st.subheader("Recent Market Data")

display_columns = [
    column
    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "RSI_14",
        "MACD",
    ]
    if column in market_data.columns
]

st.dataframe(
    market_data[
        display_columns
    ].tail(20).sort_index(ascending=False),
    use_container_width=True,
)

st.caption(
    "Market data is provided by Yahoo Finance and may be delayed or incomplete."
)
