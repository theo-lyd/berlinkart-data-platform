import pandas as pd
import numpy as np
from faker import Faker
from deltalake import write_deltalake
import os
import shutil
import time

# 1. Setup
fake = Faker('de_DE')
Faker.seed(42)
np.random.seed(42)

DATA_PATH = "/workspaces/berlinkart-data-platform/data/delta"

# Define Counts
# --- STRESS TEST CONFIGURATION ---
NUM_CUSTOMERS = 100000   # 100k Customers
NUM_PRODUCTS = 1000      # 1k Products
NUM_ORDERS = 1000000     # 1 MILLION Orders (increment from the earlier 5k orders)!

print("🚀 Starting Data Generation for BerlinKart...")
print(f"🚀 Starting Analytical/Stress Test Data Generation...")
print(f"   Target: {NUM_ORDERS:,} Orders & {NUM_CUSTOMERS:,} Customers")
start_time = time.time()

# --- 2. GENERATE CUSTOMERS (With Segments) ---
print(f"   Generating {NUM_CUSTOMERS} Customers...")

# Optimization: Use numpy for IDs to speed up generation
customer_ids = np.arange(100000, 100000 + NUM_CUSTOMERS)

# Create Segments: 10% VIPs (High Spend), 30% Regular, 60% Churned/One-time
segments = np.random.choice(['VIP', 'Regular', 'Occasional'], NUM_CUSTOMERS, p=[0.1, 0.3, 0.6])
cities_weighted = ['Berlin']*30 + ['Munich']*20 + ['Hamburg']*15 + ['Cologne']*10 + ['Frankfurt']*10 + ['Stuttgart']*15

df_customers = pd.DataFrame({
    'customer_id': customer_ids,
    'name': [fake.name() for _ in range(NUM_CUSTOMERS)],
    'email': [f"user{i}@berlinkart.de" for i in range(NUM_CUSTOMERS)],
    'city': np.random.choice(cities_weighted, NUM_CUSTOMERS), # Weighted cities
    'segment': segments,
    'signup_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(NUM_CUSTOMERS)]
})

# --- 3. GENERATE PRODUCTS (With Pareto Skew) ---
print(f"   Generating {NUM_PRODUCTS:,} Products...")
product_ids = np.arange(100, 100 + NUM_PRODUCTS)

# Pareto Principle: 20% of products get 80% of visibility weight
weights = np.random.pareto(a=1.16, size=NUM_PRODUCTS)
weights /= weights.sum() # Normalize

df_products = pd.DataFrame({
    'product_id': product_ids,
    'product_name': [f"{fake.word().capitalize()} {fake.word().capitalize()}" for _ in range(NUM_PRODUCTS)],
    'category': np.random.choice(['Electronics', 'Home', 'Fashion', 'Sports', 'Beauty'], NUM_PRODUCTS),
    'price_cents': np.random.randint(1000, 150000, NUM_PRODUCTS), # €10 - €1500
    'cost_cents': np.random.randint(500, 80000, NUM_PRODUCTS),
    'popularity_weight': weights # Helper col for order generation
})

# --- 4. GENERATE ORDERS (With Seasonality & Growth) ---
print(f"   Generating {NUM_ORDERS:,} Orders...")

# Date Generation with Seasonality (Peak in Nov/Dec)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
# Weights: Growth trend + Holiday spike
days_in_year = len(dates)
trend = np.linspace(1, 1.5, days_in_year) # 50% growth over year
seasonality = 1 + 0.5 * np.sin(np.linspace(0, 2*np.pi, days_in_year) - 2) # Sine wave
seasonality[300:330] += 2.0 # Black Friday/Xmas spike
daily_weights = trend * seasonality
daily_weights /= daily_weights.sum()

order_dates = np.random.choice(dates, NUM_ORDERS, p=daily_weights)
order_ids = [fake.unique.uuid4() for _ in range(NUM_ORDERS)]

# Assign Customers based on segment (VIPs order more)
# We simply pick random customers, but in a real sim, VIPs would appear multiple times more often.
# For speed/simplicity, we just assign randomly here, but the Seasonality is the key metric.
chosen_customers = np.random.choice(customer_ids, NUM_ORDERS)

df_orders = pd.DataFrame({
    'order_id': order_ids,
    'customer_id': chosen_customers,
    'order_date': order_dates,
    'status': np.random.choice(['completed', 'returned', 'shipped', 'placed'], NUM_ORDERS, p=[0.7, 0.1, 0.15, 0.05])
})

# --- 5. GENERATE ITEMS (Linked to Product Popularity) ---
print(f"   Generating Line Items...")
# Use the Pareto weights we calculated earlier so "Best Sellers" actually emerge
product_choices = np.random.choice(df_products['product_id'], size=NUM_ORDERS, p=df_products['popularity_weight'])

# Join to get prices
price_map = dict(zip(df_products['product_id'], df_products['price_cents']))
amounts = [price_map[pid] for pid in product_choices]

df_items = pd.DataFrame({
    'order_id': order_ids,
    'product_id': product_choices,
    'quantity': np.random.randint(1, 3, NUM_ORDERS), # 1-2 items
    'amount_cents': amounts
})

# --- 6. WRITE TO DELTA ---
tables = {"raw_customers": df_customers, "raw_products": df_products.drop(columns=['popularity_weight']), "raw_orders": df_orders, "raw_order_items": df_items}

for name, df in tables.items():
    path = os.path.join(DATA_PATH, name)
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    write_deltalake(path, df, mode="overwrite")
    print(f"   💾 {name}: {len(df):,} rows")

print(f"\n✅ Generation Complete in {time.time() - start_time:.2f}s")
print("\n✅ Data Generation Complete. All 4 tables created.")
