# 🏎️ BerlinKart Data Platform

![Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Stack](https://img.shields.io/badge/Stack-dbt%20|%20DuckDB%20|%20Docker%20|%20Streamlit-blue)
![Python](https://img.shields.io/badge/Python-3.12-yellow)

**BerlinKart** is a production-grade, local-first data platform simulating the analytics infrastructure of a high-growth e-commerce company. It demonstrates the full **Data Engineering Lifecycle**: from synthetic data generation and ingestion to dimensional modeling, testing, and advanced ML-driven analytics.

The entire stack is containerized, ensuring that the warehouse, transformations, and dashboard run consistently on any machine.

---

## 🏗️ Architecture



1.  **Ingestion Layer**: A Python generator (`ingest.py`) creates realistic e-commerce data (Customers, Products, Orders, Items) using `Faker` and `Numpy`. Data is generated with **seasonality**, **growth trends**, and **Pareto distributions** (80/20 rule) to mimic real-world retail patterns.
2.  **Storage Layer**: Raw data is stored in **Delta Lake** format for ACID compliance and versioning potential.
3.  **Warehouse Layer**: **DuckDB** serves as the high-performance, in-process analytical database. It reads directly from Delta files and materialized views.
4.  **Transformation Layer**: **dbt (data build tool)** orchestrates the ELT process:
    * **Staging**: Cleaning and casting raw data.
    * **Snapshots**: Implementing **SCD Type 2** to track customer history over time.
    * **Marts**: Building a Kimball-style **Star Schema** (`fct_orders`, `dim_customers`, `dim_products`).
5.  **Analytics Layer**: A **Streamlit** application provides an interactive Executive Dashboard featuring:
    * **Forecasting**: Holt-Winters Exponential Smoothing.
    * **Segmentation**: RFM Analysis using K-Means Clustering.
    * **Retention**: Cohort Analysis with decay curves and heatmaps.

---

## 🚀 Key Features

* **Dimensional Modeling**: Full implementation of a Star Schema with Fact and Dimension tables.
* **History Tracking**: Slowly Changing Dimensions (SCD Type 2) capture changes in customer data (e.g., address changes) without losing history.
* **Data Quality**: Automated `dbt test` suites ensure uniqueness, referential integrity (Foreign Keys), and business logic (e.g., non-negative revenue).
* **Advanced Analytics**:
    * **Predictive AI**: Forecasts revenue 12 weeks out with 95% confidence intervals.
    * **Clustering**: Automatically segments users into VIP, Loyal, and Churned groups.
    * **Cohort Matrix**: Visualizes user retention rates over time.
* **DevOps Standard**:
    * **Dockerized**: Zero-dependency setup via `docker-compose` or `docker run`.
    * **Linting**: SQLFluff configured for code style enforcement.
    * **CI/CD Ready**: Structure supports GitHub Actions for automated testing.

---

## 🛠️ Project Structure

```text
.
├── .github/              # CI/CD Workflows
├── berlinkart_dbt/       # dbt Project (The Transformation Engine)
│   ├── models/
│   │   ├── staging/      # View models on top of raw sources
│   │   └── marts/        # Final Star Schema tables (fct/dim)
│   ├── snapshots/        # SCD Type 2 definitions
│   ├── macros/           # Jinja functions (e.g., cents_to_euros)
│   └── seeds/            # Static reference data (Date Dimension)
├── data/delta/           # Raw storage (generated parquet/delta files)
├── dashboard.py          # Streamlit Analytics Application
├── ingest.py             # Synthetic Data Generator
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md             # You are here

```

---

## 💻 Getting Started

### Prerequisites

* Docker Desktop **OR** Python 3.10+ (if running locally)

### Option A: Run via Docker (Recommended)

The easiest way to view the platform. This builds the environment, generates fresh data, runs the models, and launches the app.

1. **Build the Image**:
```bash
docker build -t berlinkart:latest .

```


2. **Run the Container**:
```bash
docker run -p 8501:8501 berlinkart:latest

```


3. **Access**: Open `http://localhost:8501` in your browser.

### Option B: Run Locally (For Development)

1. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


2. **Generate Data**:
```bash
python ingest.py

```


3. **Run Transformations**:
```bash
cd berlinkart_dbt
dbt deps
dbt snapshot
dbt run
dbt test
cd ..

```


4. **Launch Dashboard**:
```bash
streamlit run dashboard.py

```



---

## 📊 Dashboard Guide

* **Trends Tab**: View monthly revenue and order volume. Download clean CSVs for offline analysis.
* **Forecasting Tab**: Toggle between **Linear Regression** (simple trend) and **Holt-Winters** (seasonality-aware) to predict Q4 sales spikes.
* **Products Tab**: Explore the **Pareto (80/20)** distribution of your product catalog.
* **RFM Tab**: Analyze how customers move between segments (e.g., "New" -> "VIP") using the Transition Matrix.
* **Cohorts Tab**: Interactive heatmap showing user retention. Click cells to drill down into specific user lists.

---

## 🔧 Technologies Used

* **Language**: Python 3.12
* **Orchestration**: dbt Core 1.8
* **Database**: DuckDB (In-process OLAP)
* **Storage**: Delta Lake
* **Visualization**: Streamlit, Altair, Plotly
* **Machine Learning**: Scikit-Learn, Statsmodels
* **Containerization**: Docker

---

*Built with ❤️ by theo_lyd as part of the BerlinKart Engineering Initiative.*
