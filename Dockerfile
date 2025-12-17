# 1. Base Image: Start with a lightweight Python version
FROM python:3.12-slim

# 2. System Setup: Install git (needed for dbt deps)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# 3. Work Directory: Where our code will live inside the container
WORKDIR /app

# 4. Install Python Dependencies
# We copy just the requirements first to cache them (speeds up future builds)
COPY requirements.txt .

# Upgrade pip first to avoid issues
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Code: Move your entire project into the container
COPY . .

# 6. Environment Variables
# Tell dbt where to find profiles.yml (we will copy it in, or rely on local)
ENV DBT_PROFILES_DIR=/app/berlinkart_dbt

# 7. Default Command
# When the container starts, it will run the dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
