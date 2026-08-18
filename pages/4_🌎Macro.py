import streamlit as st
import plotly.express as px

from utils.macro_data import get_fred_series
treasury_2y = get_fred_series("DGS2")
treasury_10y = get_fred_series("DGS10")
gdp = get_fred_series("GDPC1")

gdp["GDP Growth"] = (
    gdp["GDPC1"].pct_change(4) * 100
)

yield_data = treasury_2y.join(
    treasury_10y,
    how="inner",
)

yield_data["Spread"] = (
    yield_data["DGS10"] - yield_data["DGS2"]
)

st.set_page_config(
    page_title="Macro | Sigma Terminal",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 Macroeconomic Dashboard")

fed_funds = get_fred_series("FEDFUNDS")
unemployment = get_fred_series("UNRATE")

cpi = get_fred_series("CPIAUCSL")

cpi["Inflation"] = (
    cpi["CPIAUCSL"].pct_change(12) * 100
)

st.subheader("Federal Funds Rate")

fig_fed = px.line(
    fed_funds,
    x=fed_funds.index,
    y="FEDFUNDS",
    title="Federal Funds Rate",
)

st.plotly_chart(
    fig_fed,
    use_container_width=True,
    key="fed_funds_chart",
)

st.subheader("Unemployment Rate")

fig_unemployment = px.line(
    unemployment,
    x=unemployment.index,
    y="UNRATE",
    title="U.S. Unemployment Rate",
)

st.plotly_chart(
    fig_unemployment,
    use_container_width=True,
    key="unemployment_chart",
)

st.subheader("Inflation")

latest_inflation = cpi["Inflation"].dropna().iloc[-1]

st.metric(
    "Current YoY Inflation",
    f"{latest_inflation:.2f}%",
)

fig_inflation = px.line(
    cpi,
    x=cpi.index,
    y="Inflation",
    title="U.S. Year-over-Year Inflation",
)

fig_inflation.update_yaxes(
    title="Inflation Rate (%)"
)

st.plotly_chart(
    fig_inflation,
    use_container_width=True,
    key="inflation_chart",
)

st.subheader("Treasury Yield Analysis")

latest_2y = yield_data["DGS2"].dropna().iloc[-1]
latest_10y = yield_data["DGS10"].dropna().iloc[-1]
latest_spread = yield_data["Spread"].dropna().iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric(
    "2-Year Treasury",
    f"{latest_2y:.2f}%",
)

col2.metric(
    "10-Year Treasury",
    f"{latest_10y:.2f}%",
)

col3.metric(
    "10Y - 2Y Spread",
    f"{latest_spread:.2f}%",
)

fig_spread = px.line(
    yield_data,
    x=yield_data.index,
    y="Spread",
    title="10Y - 2Y Treasury Spread",
)

fig_spread.update_yaxes(
    title="Spread (%)"
)

st.plotly_chart(
    fig_spread,
    use_container_width=True,
    key="treasury_spread_chart",
)

st.subheader("Economic Growth")

latest_gdp_growth = gdp["GDP Growth"].dropna().iloc[-1]

st.metric(
    "Real GDP Growth (YoY)",
    f"{latest_gdp_growth:.2f}%",
)

fig_gdp = px.line(
    gdp,
    x=gdp.index,
    y="GDP Growth",
    title="U.S. Real GDP Growth",
)

fig_gdp.update_yaxes(
    title="YoY Growth (%)"
)

st.plotly_chart(
    fig_gdp,
    use_container_width=True,
    key="gdp_growth_chart",
)
