from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# 1. Define the DAG context
with DAG(
    dag_id="hello_berlinkart",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None, # Trigger manually for now
    catchup=False,
    tags=['berlinkart']
) as dag:

    # 2. Define Tasks
    task_1 = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Hello BerlinKart! The pipeline is alive!'"
    )

    task_2 = BashOperator(
        task_id="check_python",
        bash_command="python --version"
    )

    # 3. Define Dependency (The Arrow)
    task_1 >> task_2