from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# 1. Define Paths (Codespaces specific)
# We use absolute paths to avoid "File not found" errors
PROJECT_DIR = "/workspaces/berlinkart-data-platform"
DBT_DIR = f"{PROJECT_DIR}/berlinkart_dbt"
# This magic string activates your virtual environment before running a command
VENV_ACTIVATE = f"source {PROJECT_DIR}/.venv/bin/activate"

# 2. Define the DAG
with DAG(
    dag_id="berlinkart_daily_etl",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily", # Runs once every 24 hours
    catchup=False,
    tags=['production', 'berlinkart']
) as dag:

    # Task 1: Ingest Data (Python Script)
    # We combine commands using '&&'
    ingest = BashOperator(
        task_id="ingest_raw_data",
        bash_command=f"{VENV_ACTIVATE} && python {PROJECT_DIR}/generate_raw_data.py"
    )

    # Task 2: Transform Data (dbt run)
    # We must 'cd' into the dbt folder so dbt can find dbt_project.yml
    transform = BashOperator(
        task_id="dbt_run",
        bash_command=f"{VENV_ACTIVATE} && cd {DBT_DIR} && dbt run"
    )

    # Task 3: Quality Check (dbt test)
    test = BashOperator(
        task_id="dbt_test",
        bash_command=f"{VENV_ACTIVATE} && cd {DBT_DIR} && dbt test"
    )

    # 3. Define Dependencies
    ingest >> transform >> test