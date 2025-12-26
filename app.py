import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Education Indicators Dashboard")

@st.cache_data
def load_raw(path: str) -> pd.DataFrame:
    # low_memory helps with mixed types in big csvs
    return pd.read_csv(path, low_memory=True)

df_raw = load_raw("EdStatsData.csv")

# Detect format
year_cols = [c for c in df_raw.columns if str(c).strip().isdigit()]
is_wide = len(year_cols) > 0

# Basic sanity checks
if is_wide:
    needed = ["Country Name", "Indicator Name"]
    miss = [c for c in needed if c not in df_raw.columns]
    if miss:
        st.error(f"Missing columns: {miss}\nFound columns: {list(df_raw.columns)[:30]}")
        st.stop()
else:
    needed = ["Country Name", "Indicator Name", "Year", "Value"]
    miss = [c for c in needed if c not in df_raw.columns]
    if miss:
        st.error(f"Missing columns: {miss}\nFound columns: {list(df_raw.columns)[:30]}")
        st.stop()

# -------- Filters (FAST) --------
col1, col2, col3 = st.columns([2, 4, 3])

with col1:
    country = st.selectbox("Country", sorted(df_raw["Country Name"].dropna().unique()))

with col2:
    # show fewer first for speed (still searchable)
    indicator = st.selectbox("Indicator", sorted(df_raw["Indicator Name"].dropna().unique()))

with col3:
    if is_wide:
        y_min = int(min(map(int, year_cols)))
        y_max = int(max(map(int, year_cols)))
    else:
        y_min = int(pd.to_numeric(df_raw["Year"], errors="coerce").min())
        y_max = int(pd.to_numeric(df_raw["Year"], errors="coerce").max())

    start_default = max(1970, y_min)
    end_default = min(2015, y_max)
    yr = st.slider("Year range", y_min, y_max, (start_default, end_default))

# -------- Build ONLY the selected slice --------
sub = df_raw[(df_raw["Country Name"] == country) & (df_raw["Indicator Name"] == indicator)].copy()

if is_wide:
    # melt only the year columns we need
    selected_year_cols = [str(y) for y in range(yr[0], yr[1] + 1) if str(y) in sub.columns]
    if not selected_year_cols:
        st.warning("No year columns found in this range for the selected country/indicator.")
        st.stop()

    sub_long = sub.melt(
        id_vars=[c for c in sub.columns if c not in year_cols],
        value_vars=selected_year_cols,
        var_name="Year",
        value_name="Value"
    )
    sub_long["Year"] = pd.to_numeric(sub_long["Year"], errors="coerce")
    sub_long["Value"] = pd.to_numeric(sub_long["Value"], errors="coerce")
else:
    sub_long = sub.copy()
    sub_long["Year"] = pd.to_numeric(sub_long["Year"], errors="coerce")
    sub_long["Value"] = pd.to_numeric(sub_long["Value"], errors="coerce")
    sub_long = sub_long[(sub_long["Year"] >= yr[0]) & (sub_long["Year"] <= yr[1])]

sub_long = sub_long.dropna(subset=["Year"]).sort_values("Year")

# -------- 4 Visualizations (one screen) --------
left, right = st.columns(2)

# 1) Actual trend
c1 = px.line(sub_long, x="Year", y="Value", markers=True,
             title=f"Actual Trend: {indicator} — {country}")
left.plotly_chart(c1, use_container_width=True)

# 2) Regression line (no sklearn)
clean = sub_long.dropna(subset=["Value"])
if len(clean) >= 2:
    x = clean["Year"].values
    y = clean["Value"].values
    m, b = np.polyfit(x, y, 1)

    future_years = np.arange(int(clean["Year"].max()) + 1, int(clean["Year"].max()) + 11)
    yhat = m * future_years + b
    pred_df = pd.DataFrame({"Year": future_years, "Predicted": yhat})

    c2 = px.line(pred_df, x="Year", y="Predicted",
                 title="Predicted Trend (Linear Regression)")
    right.plotly_chart(c2, use_container_width=True)
else:
    right.info("Not enough data points for regression. Choose another indicator or widen the year range.")

# 3) Missing data by year (for the selected slice)
counts = sub_long.groupby("Year")["Value"].agg(
    total="size",
    missing=lambda s: s.isna().sum()
).reset_index()
counts["missing_rate"] = np.where(counts["total"] > 0, counts["missing"] / counts["total"], 0)

left2, right2 = st.columns(2)
c3 = px.bar(counts, x="Year", y="missing_rate",
            title="Missing Rate by Year",
            labels={"missing_rate": "Missing rate (0–1)"})
left2.plotly_chart(c3, use_container_width=True)

# 4) Distribution
c4 = px.histogram(clean, x="Value", nbins=25, title="Value Distribution (Non-missing)")
right2.plotly_chart(c4, use_container_width=True)

st.caption("Hover + zoom + pan are available from the Plotly toolbar on each chart.")
