import streamlit as st
import plotly.express as px

from utils.macro_data import get_fred_series


st.set_page_config(
    page_title="Macro | Sigma Terminal",
    page_icon="🌎",
    layout="wide",
)

st.title("🌎 Macroeconomic Dashboard")

fed_funds = get_fred_series("FEDFUNDS")
unemployment = get_fred_series("UNRATE")

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