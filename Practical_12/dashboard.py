
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Customer Sales Dashboard",
                   page_icon="📊", layout="wide")

st.title("Customer Sales Dashboard")
st.caption("BDA Practical 12 | Interactive Analytics")

# Load dataset
file_path = Path(__file__).parent / "sample_-_superstore.xls"

@st.cache_data
def load_data(path):
    return pd.read_excel(path)

try:
    df = load_data(str(file_path))
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.info("Check that xlrd is installed and the file is valid.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

# Check required columns
required = ["Order ID", "Order Date", "Sales", "Profit",
            "Region", "Category", "Customer ID"]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Data cleaning and validation
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

invalid_rows = df[["Order ID", "Order Date", "Sales", "Profit"]].isna().any(axis=1).sum()
df = df.dropna(subset=["Order ID", "Order Date", "Sales", "Profit"]).copy()

for col in ["Region", "Category", "Customer ID"]:
    df[col] = df[col].fillna("Unknown").astype(str)

st.subheader("Dataset Validation")
a, b, c = st.columns(3)
a.metric("Valid Transaction Rows", f"{len(df):,}")
b.metric("Invalid Rows Removed", f"{invalid_rows:,}")
c.metric("Distinct Orders", f"{df['Order ID'].nunique():,}")

# Sidebar filters
st.sidebar.header("Dashboard Filters")

regions = sorted(df["Region"].unique().tolist())
categories = sorted(df["Category"].unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Select Region", regions, default=regions
)
selected_categories = st.sidebar.multiselect(
    "Select Category", categories, default=categories
)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df[
    df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
].copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        filtered["Order Date"].dt.date.between(start, end)
    ]
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    filtered = filtered[filtered["Order Date"].dt.date == date_range[0]]
else:
    filtered = filtered[filtered["Order Date"].dt.date == date_range]

st.subheader("Key Performance Indicators")

if filtered.empty:
    st.warning("No data matches these filters. Change your selections.")
    st.stop()

sales = filtered["Sales"].sum()
profit = filtered["Profit"].sum()
orders = filtered["Order ID"].nunique()
customers = filtered["Customer ID"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"${sales:,.2f}")
k2.metric("Total Profit", f"${profit:,.2f}")
k3.metric("Total Orders", f"{orders:,}")
k4.metric("Total Customers", f"{customers:,}")

# Sales charts
st.subheader("Sales Analysis")
left, right = st.columns(2)

category_sales = filtered.groupby("Category", as_index=False)["Sales"].sum()
fig = px.bar(category_sales, x="Category", y="Sales",
             title="Sales by Category", text_auto=".2s")
left.plotly_chart(fig, use_container_width=True)

monthly = filtered.copy()
monthly["Month"] = monthly["Order Date"].dt.to_period("M").astype(str)
monthly_sales = monthly.groupby("Month", as_index=False)["Sales"].sum()
fig = px.line(monthly_sales, x="Month", y="Sales",
              title="Monthly Sales Trend", markers=True)
right.plotly_chart(fig, use_container_width=True)

# Regional profitability and customer segmentation
left, right = st.columns(2)

region_profit = filtered.groupby("Region", as_index=False)["Profit"].sum()
fig = px.bar(region_profit, x="Region", y="Profit",
             title="Profit by Region", text_auto=".2s")
left.plotly_chart(fig, use_container_width=True)

if "Segment" in filtered.columns:
    segment_sales = filtered.groupby("Segment", as_index=False)["Sales"].sum()
    fig = px.pie(segment_sales, names="Segment", values="Sales",
                 title="Sales by Customer Segment", hole=0.4)
    right.plotly_chart(fig, use_container_width=True)

# Drill-down analysis
st.subheader("Product Profitability and Drill-Down")

if "Sub-Category" in filtered.columns and "Product Name" in filtered.columns:
    level = st.selectbox(
        "Select Analysis Level",
        ["Category", "Sub-Category", "Product Name"]
    )
    product_profit = filtered.groupby(level, as_index=False)["Profit"].sum()
    product_profit = product_profit.sort_values("Profit", ascending=False)

    if level == "Product Name":
        product_profit = product_profit.head(10)

    fig = px.bar(product_profit, x="Profit", y=level, orientation="h",
                 title=f"Profit by {level}")
    st.plotly_chart(fig, use_container_width=True)

# Quarterly sales comparison
st.subheader("Quarterly Sales Performance")

quarterly = filtered.copy()
quarterly["Year"] = quarterly["Order Date"].dt.year
quarterly["Quarter"] = "Q" + quarterly["Order Date"].dt.quarter.astype(str)
quarterly["Period"] = quarterly["Year"].astype(str) + " " + quarterly["Quarter"]

quarterly_sales = quarterly.groupby(
    ["Year", "Quarter", "Period"], as_index=False
)["Sales"].sum()

fig = px.bar(quarterly_sales, x="Period", y="Sales",
             title="Sales by Year and Quarter", text_auto=".2s")
st.plotly_chart(fig, use_container_width=True)

# Transaction details and data export
with st.expander("View Filtered Transaction Data"):
    st.dataframe(filtered, use_container_width=True)

st.download_button(
    "Download Filtered Data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully.")
