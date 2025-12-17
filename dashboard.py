"""
BerlinKart Executive AI
Features:
- Materialized cohort table in DuckDB (v_cohort_retention)
- Click-to-drill Plotly heatmap (with streamlit-plotly-events fallback)
- Cohort decay curves, retention heatmap + table + drilldown consistent
- Forecast (Linear & Holt-Winters) with approximate confidence intervals
- RFM segmentation + transition matrix between two temporal slices
- Saved views via URL params (filters persist/shareable)
- "Download all datasets" ZIP bundle
- Query cost optimization: heavy work pushed into DuckDB, indexes created
- PRAGMA enable_object_cache set for DuckDB performance
"""
import os
import io
import zipfile
import importlib
from datetime import datetime
from typing import Optional
import time

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional dependency for capturing Plotly click events
PLOTLY_EVENTS_AVAILABLE = importlib.util.find_spec("streamlit_plotly_events") is not None
if PLOTLY_EVENTS_AVAILABLE:
    from streamlit_plotly_events import plotly_events

# --- CONFIG ---
st.set_page_config(page_title="BerlinKart Executive AI", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; color: #0068c9; }
div.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- HELPERS: exports & small utils ---
def format_currency(value):
    try:
        value = float(value)
    except Exception:
        return "€0"
    if value >= 1_000_000_000:
        return f"€{value/1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"€{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"€{value/1_000:.0f}K"
    return f"€{value:,.2f}"

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    # Try engines, fallback to ZIP-of-CSV
    for engine in ("xlsxwriter", "openpyxl"):
        if importlib.util.find_spec(engine) is not None:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
            bio.seek(0)
            return bio.read()
    # fallback -> ZIP containing CSV
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{sheet_name.replace(' ','_').lower()}.csv", df.to_csv(index=False))
    bio.seek(0)
    return bio.read()

def safe_download_buttons(df: pd.DataFrame, name_prefix: str):
    if df is None or df.empty:
        return
    st.download_button(
        label=f"Download {name_prefix} (CSV)",
        data=df_to_csv_bytes(df),
        file_name=f"{name_prefix}.csv",
        mime="text/csv"
    )
    try:
        xlsx_bytes = df_to_excel_bytes(df, sheet_name=name_prefix[:31])
        # detect ZIP fallback by magic bytes
        if xlsx_bytes[:2] == b'PK':
            st.download_button(
                label=f"Download {name_prefix} (ZIP of CSV)",
                data=xlsx_bytes,
                file_name=f"{name_prefix}.zip",
                mime="application/zip"
            )
            st.info("Excel engine not available; delivered ZIP containing CSV.")
        else:
            st.download_button(
                label=f"Download {name_prefix} (Excel)",
                data=xlsx_bytes,
                file_name=f"{name_prefix}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Export failed: {e}")

def zip_multiple_dfs(df_dict: dict) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in df_dict.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    bio.seek(0)
    return bio.read()

# --- DB connection (FIXED: Includes View Registration) ---
@st.cache_resource(show_spinner=False)
def get_con(path="berlinkart_dbt/dev.duckdb"):
    con = duckdb.connect(path, read_only=False)
    # performance pragma
    try:
        con.execute("PRAGMA enable_object_cache=TRUE;")
    except Exception:
        pass
    # Ensure delta extension loaded
    try:
        con.execute("INSTALL delta; LOAD delta;")
    except Exception:
        pass

    # --- CRITICAL FIX: Register Raw Files as Views ---
    # This ensures DuckDB can see the data in data/delta/
    tables = ['raw_order_items', 'raw_products', 'raw_customers', 'raw_orders']
    for t in tables:
        file_path = f"data/delta/{t}"
        if os.path.exists(file_path):
            try:
                con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM delta_scan('{file_path}')")
            except Exception as e:
                print(f"Warning: Failed to register view {t}: {e}")
    # ------------------------------------------------

    return con

con = get_con()

# --- URL params: saved views (read & write) ---
def load_params_or_defaults():
    params = st.query_params  # <-- replace experimental_get_query_params
    # defaults fetched from DB
    try:
        min_date, max_date = con.sql("SELECT min(order_date), max(order_date) FROM fct_orders").fetchone()
    except Exception:
        min_date, max_date = None, None

    # param values (strings) fall back to DB defaults; convert to date as needed
    if params.get("start"):
        start = pd.to_datetime(params.get("start")).date()
    else:
        start = pd.to_datetime(min_date).date() if min_date is not None else None

    if params.get("end"):
        end = pd.to_datetime(params.get("end")).date()
    else:
        end = pd.to_datetime(max_date).date() if max_date is not None else None

    # streamlit returns string or list depending on version, normalize to list
    cities = params.get_all("cities") if hasattr(params, "get_all") else (params.get("cities") if isinstance(params.get("cities"), list) else [])

    return start, end, cities

start_default, end_default, cities_default = load_params_or_defaults()

# --- SIDEBAR ---
st.sidebar.title("Controls")
# date_input can accept date objects
start_date, end_date = st.sidebar.date_input("Analysis window", value=[start_default, end_default], min_value=start_default, max_value=end_default)

all_cities = [r[0] for r in con.sql("SELECT DISTINCT city FROM dim_customers ORDER BY 1").fetchall()]
# Handle case where cities_default might be None or empty
default_selection = cities_default if cities_default else all_cities[:5]
# Ensure default values are actually in options
default_selection = [c for c in default_selection if c in all_cities]

selected_cities = st.sidebar.multiselect("Cities", options=all_cities, default=default_selection)

# Save view (URL)
if st.sidebar.button("Save view to URL"):
    params = {}
    params["start"] = start_date.isoformat()
    params["end"] = end_date.isoformat()
    if selected_cities:
        params["cities"] = selected_cities
    st.query_params.update(params)
    st.sidebar.success("Saved to URL — copy the URL to share the view.")

# Quick controls for modeling
st.sidebar.markdown("### Forecast controls")
default_method = st.sidebar.selectbox("Forecast method", ["Linear (monthly, 3m)", "Holt-Winters (weekly)"])

# Build SQL city filter safely
if selected_cities:
    safe_cities = "', '".join([c.replace("'", "''") for c in selected_cities])
    city_filter = f"c.city IN ('{safe_cities}')"
else:
    city_filter = "1=1"

# --- Heavy aggregation moved into DuckDB ---
import time

@st.cache_data(ttl=300, show_spinner=False)
def build_cohort_materialized(start_date, end_date, city_filter):
    con_local = get_con()   # resource-cached connection

    drop_v = "DROP TABLE IF EXISTS v_cohort_retention;"
    create_v = f"""
    CREATE TABLE v_cohort_retention AS
    WITH customer_first AS (
        SELECT
            f.customer_id AS customer_id,
            date_trunc('month', min(f.order_date))::DATE AS cohort_month
        FROM fct_orders f
        JOIN dim_customers c
        ON f.customer_id = c.customer_id
        WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        AND {city_filter}
        GROUP BY f.customer_id
    ),
    order_activity AS (
        SELECT
            f.customer_id AS customer_id,
            date_trunc('month', f.order_date)::DATE AS activity_month
        FROM fct_orders f
        JOIN dim_customers c
        ON f.customer_id = c.customer_id
        WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        AND {city_filter}
    ),
    cohort_counts AS (
        SELECT
            cf.cohort_month,
            strftime(cf.cohort_month, '%Y-%m') AS cohort_label,
            DATE_DIFF('month', cf.cohort_month, oa.activity_month) AS month_index,
            COUNT(DISTINCT cf.customer_id) FILTER (
                WHERE DATE_DIFF('month', cf.cohort_month, oa.activity_month) BETWEEN 0 AND 60
            ) AS users
        FROM customer_first cf
        LEFT JOIN order_activity oa
        ON cf.customer_id = oa.customer_id
        GROUP BY cf.cohort_month, cohort_label, month_index
    )
    SELECT
        cohort_label AS cohort,
        cohort_month,
        COALESCE(month_index, 0) AS month_index,
        COALESCE(users, 0) AS users
    FROM cohort_counts
    ORDER BY cohort_month DESC, month_index;
    """

    drop_stats = "DROP TABLE IF EXISTS v_cohort_retention_stats;"
    create_stats = """
    CREATE TABLE v_cohort_retention_stats AS
    SELECT cohort, cohort_month, SUM(users) FILTER (WHERE month_index = 0) AS cohort_size
    FROM v_cohort_retention
    GROUP BY cohort, cohort_month;
    """

    drop_view = "DROP VIEW IF EXISTS v_cohort_retention_view;"
    create_view = """
    CREATE VIEW v_cohort_retention_view AS
    SELECT r.cohort, r.cohort_month, r.month_index, r.users,
           s.cohort_size,
           CASE WHEN s.cohort_size = 0 THEN 0.0 ELSE r.users::DOUBLE / s.cohort_size END AS retention_rate
    FROM v_cohort_retention r
    LEFT JOIN v_cohort_retention_stats s ON r.cohort = s.cohort AND r.cohort_month = s.cohort_month
    ORDER BY r.cohort_month DESC, r.month_index;
    """

    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            con_local.execute(drop_v)
            con_local.execute(create_v)
            try:
                con_local.execute("CREATE INDEX IF NOT EXISTS idx_v_cohort_month ON v_cohort_retention (cohort_month);")
                con_local.execute("CREATE INDEX IF NOT EXISTS idx_v_cohort_month_index ON v_cohort_retention (cohort_month, month_index);")
            except Exception:
                pass
            con_local.execute(drop_stats)
            con_local.execute(create_stats)
            con_local.execute(drop_view)
            con_local.execute(create_view)
            break
        except Exception as e:
            msg = str(e).lower()
            if "transaction" in msg:
                if attempt < attempts:
                    time.sleep(0.25 * attempt)
                    continue
                else:
                    raise
            else:
                raise

    df = con_local.sql(
        "SELECT cohort, cohort_month, month_index, users, cohort_size, retention_rate "
        "FROM v_cohort_retention_view ORDER BY cohort_month DESC, month_index"
    ).df()
    return df

cohort_df = build_cohort_materialized(start_date, end_date, city_filter)

if cohort_df is None or cohort_df.empty:
    cohort_df = pd.DataFrame(columns=["cohort","cohort_month","month_index","users","cohort_size","retention_rate"])

# --- KPI banner ---
st.title("🇩🇪 BerlinKart Executive AI — Final")
st.caption(f"Analysis window: {start_date} → {end_date} — Cities: {', '.join(selected_cities) if selected_cities else 'All'}")
kpi = con.sql(f"""
    SELECT COUNT(DISTINCT f.order_id) AS orders,
           SUM(f.revenue_euro) AS revenue,
           AVG(f.revenue_euro) AS aov,
           COUNT(DISTINCT f.customer_id) AS active_customers
    FROM fct_orders f
    JOIN dim_customers c ON f.customer_id = c.customer_id
    WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
      AND {city_filter}
""").df()
orders_val = int(kpi['orders'][0]) if not kpi.empty and pd.notna(kpi['orders'][0]) else 0
revenue_val = float(kpi['revenue'][0]) if not kpi.empty and pd.notna(kpi['revenue'][0]) else 0.0
aov_val = float(kpi['aov'][0]) if not kpi.empty and pd.notna(kpi['aov'][0]) else 0.0
active_customers_val = int(kpi['active_customers'][0]) if not kpi.empty and pd.notna(kpi['active_customers'][0]) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", format_currency(revenue_val))
c2.metric("Total Orders", f"{orders_val:,}")
c3.metric("AOV", f"€{aov_val:.2f}")
c4.metric("Active Customers", f"{active_customers_val:,}")

st.divider()

# --- TABS ---
tab_trends, tab_forecast, tab_products, tab_rfm, tab_cohort = st.tabs([
    "Performance Trends", "Forecasting", "Products", "RFM & Transitions", "Cohorts (heatmap + drill)"
])

# --- Trends tab ---
with tab_trends:
    st.subheader("Monthly Revenue & Orders")
    trend_df = con.sql(f"""
        SELECT date_trunc('month', f.order_date)::DATE AS month,
               SUM(f.revenue_euro) AS revenue,
               COUNT(f.order_id) AS orders
        FROM fct_orders f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
          AND {city_filter}
        GROUP BY month
        ORDER BY month
    """).df()
    if trend_df.empty:
        st.warning("No trend data for selection.")
    else:
        line_rev = alt.Chart(trend_df).mark_line(color='#0f54c9').encode(x='month:T', y=alt.Y('revenue:Q', axis=alt.Axis(format='~s')), tooltip=['month','revenue'])
        line_orders = alt.Chart(trend_df).mark_line(strokeDash=[5,5], color='#ff8c00').encode(x='month:T', y='orders:Q', tooltip=['month','orders'])
        st.altair_chart((line_rev + line_orders).resolve_scale(y='independent'), use_container_width=True)
        safe_download_buttons(trend_df, "monthly_trend")

# --- Forecasting tab ---
with tab_forecast:
    st.subheader("Forecasting — with approximate confidence intervals")
    method = st.selectbox("Method", ["Linear Regression (monthly)", "Holt-Winters (weekly)"], index=0 if default_method.startswith("Linear") else 1)

    if method.startswith("Linear"):
        df_ml = trend_df.copy()
        if len(df_ml) < 3:
            st.warning("Need >= 3 monthly points for regression forecasting.")
        else:
            df_ml['ordinal'] = pd.to_datetime(df_ml['month']).map(pd.Timestamp.toordinal)
            X = df_ml[['ordinal']].values
            y = df_ml['revenue'].values
            lr = LinearRegression()
            lr.fit(X, y)
            last_date = pd.to_datetime(df_ml['month'].max())
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
            future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            preds = lr.predict(future_ord)
            resid = y - lr.predict(X)
            se = resid.std(ddof=1)
            z = 1.96
            lower = preds - z * se
            upper = preds + z * se
            forecast_df = pd.DataFrame({'time': future_dates, 'revenue': preds, 'lower': lower, 'upper': upper, 'type': 'Forecast'})
            actual_df = df_ml.rename(columns={'month':'time'})[['time','revenue']].assign(type='Actual')
            combined = pd.concat([actual_df, forecast_df[['time','revenue','lower','upper','type']]], ignore_index=True)
            base = alt.Chart(combined).encode(x='time:T')
            actual_line = base.transform_filter(alt.datum.type == 'Actual').mark_line(color='#0f54c9').encode(y='revenue:Q')
            forecast_line = base.transform_filter(alt.datum.type == 'Forecast').mark_line(color='#ff8c00').encode(y='revenue:Q')
            band = alt.Chart(forecast_df).mark_area(opacity=0.2, color='#ff8c00').encode(x='time:T', y='lower:Q', y2='upper:Q')
            st.altair_chart((actual_line + forecast_line + band).interactive(), use_container_width=True)
            safe_download_buttons(combined.reset_index(drop=True), "lr_forecast")
    else:
        ts_df = con.sql(f"""
            SELECT date_trunc('week', f.order_date)::DATE AS week,
                   SUM(f.revenue_euro) AS revenue
            FROM fct_orders f
            JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
              AND {city_filter}
            GROUP BY week
            ORDER BY week
        """).df()
        if ts_df.empty or len(ts_df) < 12:
            st.warning("Need >= 12 weekly points for meaningful Holt-Winters.")
        else:
            ts_df = ts_df.set_index('week')
            ts_df.index = pd.to_datetime(ts_df.index)
            n = len(ts_df)
            default_sp = 52 if n >= 52 else (13 if n >= 26 else (12 if n >= 12 else None))
            sp_input = st.number_input(f"seasonal_periods (auto:{default_sp}) — set 0 to disable", min_value=0, max_value=260, value=(default_sp if default_sp else 0))
            season = int(sp_input) if sp_input > 0 else None
            fh = st.slider("Forecast horizon (weeks)", min_value=4, max_value=52, value=12)
            try:
                if season and season >= 2 and season < len(ts_df):
                    model = ExponentialSmoothing(ts_df['revenue'], trend='add', seasonal='add', seasonal_periods=season).fit()
                else:
                    model = ExponentialSmoothing(ts_df['revenue'], trend='add', seasonal=None).fit()
                forecast_values = model.forecast(fh)
                fitted = model.fittedvalues
                resid = ts_df['revenue'] - fitted
                se = float(np.nanstd(resid))
                z = 1.96
                lower = forecast_values - z * se
                upper = forecast_values + z * se
                history = ts_df.reset_index().rename(columns={'week':'time'})
                history['Type'] = 'Actual'
                future = pd.DataFrame({'time': forecast_values.index, 'revenue': forecast_values.values, 'lower': lower.values, 'upper': upper.values, 'Type': 'Forecast'})
                combined = pd.concat([history.rename(columns={'revenue':'revenue'}).assign(lower=np.nan, upper=np.nan).rename(columns={'Type':'Type'}), future], ignore_index=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history['time'], y=history['revenue'], name='Actual', line=dict(color='#0f54c9')))
                fig.add_trace(go.Scatter(x=future['time'], y=future['revenue'], name='Forecast', line=dict(color='#ff8c00')))
                fig.add_trace(go.Scatter(x=list(future['time']) + list(future['time'][::-1]),
                                         y=list(future['upper']) + list(future['lower'][::-1]),
                                         fill='toself', fillcolor='rgba(255,140,0,0.2)', line=dict(color='rgba(255,140,0,0)'),
                                         hoverinfo="skip", showlegend=True, name='95% CI'))
                st.plotly_chart(fig, use_container_width=True)
                safe_download_buttons(combined.reset_index(drop=True), "holtwinters_forecast")
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# --- Products tab ---
with tab_products:
    st.subheader("Top products (Pareto)")
    # Raw items view now guaranteed by get_con()
    try:
        prod_df = con.sql(f"""
            SELECT p.product_name, p.category, SUM(i.amount_cents)/100.0 AS revenue
            FROM raw_order_items i
            JOIN dim_products p ON i.product_id = p.product_id
            JOIN fct_orders f ON i.order_id = f.order_id
            JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE f.order_date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
              AND {city_filter}
            GROUP BY p.product_name, p.category
            ORDER BY revenue DESC
            LIMIT 100
        """).df()
        if prod_df.empty:
            st.warning("No product data.")
        else:
            bar = alt.Chart(prod_df).mark_bar().encode(
                x=alt.X('revenue:Q', axis=alt.Axis(format='~s')),
                y=alt.Y('product_name:N', sort='-x'),
                color='category:N',
                tooltip=['product_name','category','revenue']
            )
            st.altair_chart(bar, use_container_width=True)
            safe_download_buttons(prod_df, "top_products")
    except Exception as e:
        st.error(f"Product query failed: {e}. Try rebuilding Docker container if raw views are missing.")

# --- RFM & Transition matrices ---
with tab_rfm:
    st.subheader("RFM Segmentation and Transition Matrix")
    delta_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    split_point = pd.to_datetime(start_date) + pd.Timedelta(days=max(1, delta_days//2))
    st.info(f"Transition snapshots: Snapshot A = {start_date} → {split_point.date()}, Snapshot B = {split_point.date()} → {end_date}")

    def build_rfm_snapshot(con, start_d, end_d):
        q = f"""
            SELECT f.customer_id,
                   MAX(f.order_date)::DATE AS last_order,
                   COUNT(f.order_id) AS frequency,
                   SUM(f.revenue_euro) AS monetary
            FROM fct_orders f
            JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE f.order_date BETWEEN DATE '{start_d}' AND DATE '{end_d}'
              AND {city_filter}
            GROUP BY f.customer_id
        """
        return con.sql(q).df()

    snap_a = build_rfm_snapshot(con, start_date, split_point.date())
    snap_b = build_rfm_snapshot(con, split_point.date(), end_date)

    if snap_a.empty or snap_b.empty:
        st.warning("Not enough data in one of the snapshots to compute transitions.")
    else:
        k = st.sidebar.slider("RFM clusters (k)", 2, 8, 4)
        scaler = StandardScaler()
        # --- FIXED: Explicitly name the Index to prevent KeyError 'customer_id' ---
        customers = pd.Index(sorted(set(snap_a['customer_id']).union(set(snap_b['customer_id']))), name='customer_id')

        def prepare(rdf):
            # Reset index will now correctly create 'customer_id' column because index has a name
            rdf2 = rdf.set_index('customer_id').reindex(customers).fillna(0).reset_index()
            rdf2['recency'] = (pd.to_datetime(end_date) - pd.to_datetime(rdf2['last_order']).fillna(pd.Timestamp(end_date))).dt.days
            return rdf2[['customer_id','recency','frequency','monetary']]

        pa = prepare(snap_a)
        pb = prepare(snap_b)
        combined_data = pd.concat([pa[['recency','frequency','monetary']], pb[['recency','frequency','monetary']]], axis=0).fillna(0)
        scaler.fit(combined_data)
        Xa = scaler.transform(pa[['recency','frequency','monetary']].fillna(0))
        Xb = scaler.transform(pb[['recency','frequency','monetary']].fillna(0))
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(np.vstack([Xa, Xb]))
        pa['cluster'] = model.predict(Xa)
        pb['cluster'] = model.predict(Xb)
        trans = pd.crosstab(pa['cluster'], pb['cluster'], normalize='index')
        st.write("Transition matrix (rows = Snapshot A cluster, cols = Snapshot B cluster)")
        st.dataframe((trans * 100).round(1).astype(str) + '%', use_container_width=True)
        trans_counts = pd.crosstab(pa['cluster'], pb['cluster'])
        st.write("Transition counts")
        st.dataframe(trans_counts, use_container_width=True)
        safe_download_buttons(trans_counts.reset_index(), "rfm_transition_counts")

# --- Cohorts tab ---
with tab_cohort:
    st.subheader("Cohort retention — heatmap")
    if cohort_df.empty:
        st.warning("No cohort data for this selection.")
    else:
        cohort_pivot = cohort_df.pivot(index='cohort', columns='month_index', values='retention_rate').fillna(0)
        users_pivot = cohort_df.pivot(index='cohort', columns='month_index', values='users').fillna(0).astype(int)
        cohorts = list(cohort_pivot.index)
        months = list(map(int, cohort_pivot.columns.astype(int)))
        z = cohort_pivot.values.tolist()
        text = [[f"{users_pivot.iloc[r, c]} users\n{cohort_pivot.iloc[r, c]:.1%}" for c in range(cohort_pivot.shape[1])] for r in range(cohort_pivot.shape[0])]
        fig = go.Figure(data=go.Heatmap(z=z, x=months, y=cohorts, text=text, hoverinfo='text', colorscale='Blues', colorbar=dict(title='Retention')))
        fig.update_layout(height=600, xaxis_title='Months since signup', yaxis_title='Signup cohort (YYYY-MM)')

        st.markdown("**Click a cell on the heatmap to drill down**")
        if PLOTLY_EVENTS_AVAILABLE:
            selected_points = plotly_events(fig, click_event=True, hover_event=False)
            if selected_points:
                p = selected_points[0]
                st.session_state['drill_cohort'] = p.get('y')
                st.session_state['drill_month'] = int(p.get('x'))
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.warning("Install 'streamlit-plotly-events' for click interactions.")

        # Decay curves
        max_cohorts = st.sidebar.slider("Decay curves: number of cohorts", 3, min(20, len(cohorts)), 6)
        decay_df = cohort_df[cohort_df['cohort'].isin(cohorts[:max_cohorts])][['cohort','month_index','retention_rate']]
        line = alt.Chart(decay_df).mark_line(point=True).encode(x='month_index:Q', y='retention_rate:Q', color='cohort:N', tooltip=['cohort','retention_rate']).interactive()
        st.altair_chart(line, use_container_width=True)

        # Drilldown display
        if 'drill_cohort' in st.session_state:
            d_cohort = st.session_state['drill_cohort']
            d_month = st.session_state['drill_month']
            st.write(f"### Users in Cohort {d_cohort} / Month {d_month}")
            try:
                cohort_date = pd.to_datetime(d_cohort + "-01").date()
                target_month = (pd.to_datetime(cohort_date) + pd.DateOffset(months=d_month)).date()
                drill_sql = f"""
                    WITH target_customers AS (
                        SELECT customer_id FROM fct_orders f JOIN dim_customers c ON f.customer_id = c.customer_id
                        WHERE date_trunc('month', min(f.order_date))::DATE = DATE '{cohort_date}'
                        AND {city_filter}
                        GROUP BY customer_id
                    )
                    SELECT f.customer_id, f.order_id, f.revenue_euro, f.order_date
                    FROM fct_orders f
                    JOIN target_customers tc ON f.customer_id = tc.customer_id
                    WHERE date_trunc('month', f.order_date)::DATE = DATE '{target_month}'
                    LIMIT 1000
                """
                # Note: The query above needs aggregated grouping inside CTE to work on duckdb strict group by
                # We fix logic:
                drill_sql_fixed = f"""
                 WITH cohort_ids AS (
                    SELECT f.customer_id
                    FROM fct_orders f
                    JOIN dim_customers c ON f.customer_id=c.customer_id
                    WHERE {city_filter}
                    GROUP BY f.customer_id
                    HAVING date_trunc('month', min(f.order_date))::DATE = DATE '{cohort_date}'
                 )
                 SELECT f.customer_id, f.order_id, f.revenue_euro, f.order_date
                 FROM fct_orders f
                 JOIN cohort_ids ci ON f.customer_id = ci.customer_id
                 WHERE date_trunc('month', f.order_date)::DATE = DATE '{target_month}'
                 LIMIT 1000
                """
                users_df = con.sql(drill_sql_fixed).df()
                st.dataframe(users_df, use_container_width=True)
            except Exception as e:
                st.error(f"Drilldown failed: {e}")

# --- Download all ---
st.sidebar.markdown("---")
if st.sidebar.button("Download all prepared datasets"):
    prepared = {
        "monthly_trend": (trend_df if 'trend_df' in locals() else pd.DataFrame()),
        "cohort_materialized": cohort_df.reset_index(drop=True),
        "top_products": (prod_df if 'prod_df' in locals() else pd.DataFrame()),
    }
    zip_bytes = zip_multiple_dfs(prepared)
    st.download_button("Download ZIP", data=zip_bytes, file_name="berlinkart_all.zip", mime="application/zip")
