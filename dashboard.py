import streamlit as st
import duckdb
import pandas as pd
import altair as alt

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="BerlinKart HQ",
    page_icon="🇩🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. DATA CONNECTION (Optimized)
@st.cache_resource
def get_connection():
    """Establishes a read-only connection to the DuckDB warehouse"""
    try:
        # Use read_only=True to prevent locking the database
        con = duckdb.connect("berlinkart_dbt/dev.duckdb", read_only=True)
        return con
    except Exception as e:
        st.error(f"🚨 Connection Failed: {e}")
        st.stop()

con = get_connection()

# 3. SIDEBAR CONTROLS
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Filter the dashboard views.")

# Dynamic Date Filter (getting min/max from DB)
min_max_date = con.sql("SELECT min(order_date), max(order_date) FROM fct_orders").fetchone()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=[min_max_date[0], min_max_date[1]]
)

# 4. MAIN DASHBOARD HEADER
st.title("🇩🇪 BerlinKart Executive Dashboard")
st.markdown(f"**Data Range:** `{start_date}` to `{end_date}`")
st.divider()

# 5. KPIS (Using Efficient Aggregations)
# Note: We filter by the selected date range in the WHERE clause
kpi_query = f"""
    SELECT
        count(distinct order_id) as total_orders,
        sum(revenue_euro) as total_revenue,
        avg(revenue_euro) as aov,
        count(distinct customer_id) as active_customers
    FROM fct_orders
    WHERE order_date BETWEEN '{start_date}' AND '{end_date}'
"""
kpis = con.sql(kpi_query).df()

# Display KPIs in 4 columns
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", f"€{kpis['total_revenue'][0]:,.2f}", delta="vs prev period")
k2.metric("Total Orders", f"{kpis['total_orders'][0]:,}")
k3.metric("Avg Order Value", f"€{kpis['aov'][0]:.2f}")
k4.metric("Active Customers", f"{kpis['active_customers'][0]:,}")

st.divider()

# 6. ANALYTICS TABS
tab1, tab2, tab3 = st.tabs(["📈 Revenue Trends", "📍 Geographic Performance", "📦 Product Insights"])

# --- TAB 1: REVENUE OVER TIME ---
with tab1:
    st.subheader("Revenue Trajectory")

    trend_query = f"""
        SELECT
            date_trunc('month', order_date) as month,
            sum(revenue_euro) as revenue,
            count(order_id) as orders
        FROM fct_orders
        WHERE order_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 1
        ORDER BY 1
    """
    df_trend = con.sql(trend_query).df()

    # Altair Chart
    chart = alt.Chart(df_trend).mark_line(point=True).encode(
        x=alt.X('month', title='Month'),
        y=alt.Y('revenue', title='Revenue (€)'),
        tooltip=['month', 'revenue', 'orders']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# --- TAB 2: GEOGRAPHIC PERFORMANCE (Joins!) ---
with tab2:
    st.subheader("Top Performing Cities")

    geo_query = f"""
        SELECT
            c.city,
            sum(f.revenue_euro) as revenue
        FROM fct_orders f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 10
    """
    df_geo = con.sql(geo_query).df()

    # Bar Chart
    bar_chart = alt.Chart(df_geo).mark_bar().encode(
        x=alt.X('revenue', title='Revenue (€)'),
        y=alt.Y('city', sort='-x', title='City'),
        color=alt.value('#FF4B4B')
    )
    st.altair_chart(bar_chart, use_container_width=True)

# --- TAB 3: PRODUCT INSIGHTS ---
with tab3:
    st.subheader("Best Selling Categories")

    # Note: We need to join fct -> order_items -> products
    # But since fct_orders is aggregated, we might not have product_id directly depending on your model.
    # If fct_orders doesn't have product_id (it's 1 row per order), we can't easily break down by product
    # UNLESS we use the raw line items.
    # FOR THIS DEMO: We will infer from a joined query on raw items if available,
    # or just show Status breakdown which is available in fct_orders.

    status_query = f"""
        SELECT
            status,
            count(*) as count
        FROM fct_orders
        WHERE order_date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY 1
    """
    df_status = con.sql(status_query).df()

    st.bar_chart(df_status.set_index('status'))

# 7. FOOTER
st.markdown("---")
st.caption("🚀 BerlinKart Data Platform | Powered by dbt, DuckDB, & Streamlit")
