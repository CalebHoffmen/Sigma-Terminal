import streamlit as st

st.set_page_config(
    page_title="Σ Sigma Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📈 Σ Sigma Terminal")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Modules

- 📈 Markets
- 💼 Portfolio
- 📊 Optimization
- ⚠️ Risk Analytics
- 💹 Options
- 🌍 Macro
- 📉 Backtesting
- 🤖 AI Research
""")

st.sidebar.markdown("---")
st.sidebar.info("Version 1.0")

# -----------------------------
# Main Page
# -----------------------------

st.title("📈 Σ Sigma Terminal")

st.markdown("""
Welcome to **Σ Sigma Terminal**.

This application is designed to provide institutional-grade financial analytics,
portfolio management, quantitative research, and market intelligence.

Use the navigation sidebar to explore future modules.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Portfolio Value",
        value="$125,420",
        delta="+2.34%"
    )

with col2:
    st.metric(
        label="S&P 500",
        value="6,310",
        delta="+0.62%"
    )

with col3:
    st.metric(
        label="VIX",
        value="15.8",
        delta="-3.1%"
    )

st.divider()

st.subheader("Project Roadmap")

st.checkbox("Market Dashboard")
st.checkbox("Portfolio Manager")
st.checkbox("Portfolio Optimization")
st.checkbox("Risk Analytics")
st.checkbox("Factor Investing")
st.checkbox("Options Analytics")
st.checkbox("Macroeconomic Dashboard")
st.checkbox("AI Research Assistant")