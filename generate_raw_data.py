import pandas as pd
import numpy as np
from faker import Faker
from deltalake import write_deltalake
import os

# 1. Setup German Data Generator
fake = Faker('de_DE')  # German locale
Faker.seed(42)  # Reproducibility

# 2. Define Constants
NUM_ROWS = 10000
REGIONS = ['Berlin', 'Bayern', 'Hamburg', 'Nordrhein-Westfalen', 'Hessen']
STATUSES = ['completed', 'returned', 'shipped', 'processing']

# 3. Generate Data
print(f"Generating {NUM_ROWS} rows of synthetic German retail data...")

data = {
    'order_id': [fake.uuid4() for _ in range(NUM_ROWS)],
    'customer_id': np.random.randint(1000, 9999, size=NUM_ROWS),
    'order_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(NUM_ROWS)],
    'region': np.random.choice(REGIONS, size=NUM_ROWS),
    'amount': np.round(np.random.uniform(10.0, 500.0, size=NUM_ROWS), 2),
    'status': np.random.choice(STATUSES, size=NUM_ROWS, p=[0.6, 0.1, 0.2, 0.1]) # Weighted probabilities
}

df = pd.DataFrame(data)

# 4. Enforce Data Types (Crucial for Delta Lake strictness)
df['order_date'] = pd.to_datetime(df['order_date'])
df['customer_id'] = df['customer_id'].astype('int32')

# 5. Write to Delta Lake (Simulating Fabric OneLake)
# In Codespaces, we write to a local folder. Later we can move this to MinIO.
storage_path = "data/delta/raw_orders"

# Ensure directory is clean or handles overwrite
if not os.path.exists("data/delta"):
    os.makedirs("data/delta")

print(f"Writing Delta Table to {storage_path}...")

write_deltalake(
    storage_path,
    df,
    mode="overwrite"
)

print("✅ Success! Delta Table created.")
print(f"Verify by running: duckdb -c \"SELECT * FROM delta_scan('{storage_path}') LIMIT 5;\"")
