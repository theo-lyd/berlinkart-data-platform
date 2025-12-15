import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px

# 1. Page Setup
st.set_page_config(page_title="BerlinKart Executive Dashboard", layout="wide")
st.title("🇩🇪 BerlinKart Analytics (Local Fabric)")

# 2. Connect to the Lakehouse (DuckDB)
# read_only=True prevents locking issues while dbt might be running
conn = duckdb.connect('berlinkart_dbt/dev.duckdb', read_only=True)

# 3. Load Data (Consuming the Marts)
# Notice we join the Fact and Dimension here for the visualization
df = conn.sql("""
    SELECT
        f.order_date,
        f.amount_cents / 100.0 as amount_euro,
        f.status,
        c.region
    FROM fct_orders f
    JOIN dim_customers c ON f.customer_id = c.customer_id
""").to_df()

# 4. KPI Section
col1, col2, col3 = st.columns(3)

total_revenue = df['amount_euro'].sum()
total_orders = df.shape[0]
avg_order = df['amount_euro'].mean()

col1.metric("Total Revenue", f"€{total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Order Value", f"€{avg_order:.2f}")

st.markdown("---")

# 5. Charts Generation
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Revenue by Region")
    # Group by Region
    region_stats = df.groupby('region')['amount_euro'].sum().reset_index()
    fig_bar = px.bar(region_stats, x='region', y='amount_euro', color='region')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Orders over Time")
    # Group by Date
    daily_stats = df.groupby('order_date')['amount_euro'].sum().reset_index()
    fig_line = px.line(daily_stats, x='order_date', y='amount_euro')
    st.plotly_chart(fig_line, use_container_width=True)

# 6. Raw Data Viewer (Optional)
with st.expander("View Raw Data"):
    st.dataframe(df.head(100))
