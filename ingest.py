import pandas as pd
import numpy as np
from faker import Faker
from deltalake import write_deltalake
import os
import shutil

# 1. Setup
fake = Faker('de_DE')
Faker.seed(42)
np.random.seed(42)

DATA_PATH = "/workspaces/berlinkart-data-platform/data/delta"

# Define Counts
NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 50
NUM_ORDERS = 5000  # More orders than customers

print("🚀 Starting Data Generation for BerlinKart...")

# --- 2. GENERATE CUSTOMERS ---
print(f"   Generating {NUM_CUSTOMERS} Customers...")
customers = []
for _ in range(NUM_CUSTOMERS):
    customers.append({
        'customer_id': fake.unique.random_int(min=10000, max=99999),
        'name': fake.name(),
        'email': fake.email(),
        'city': fake.city(),
        'address': fake.street_address(),
        'signup_date': fake.date_between(start_date='-3y', end_date='-1y')
    })
df_customers = pd.DataFrame(customers)

# --- 3. GENERATE PRODUCTS ---
print(f"   Generating {NUM_PRODUCTS} Products...")
categories = ['Electronics', 'Home', 'Clothing', 'Books', 'Sports']
products = []
for _ in range(NUM_PRODUCTS):
    products.append({
        'product_id': fake.unique.random_int(min=100, max=999),
        'product_name': fake.word().capitalize() + " " + fake.word().capitalize(),
        'category': np.random.choice(categories),
        'price_cents': np.random.randint(500, 50000), # 5.00 to 500.00 Euro
        'cost_cents': np.random.randint(100, 20000)
    })
df_products = pd.DataFrame(products)

# --- 4. GENERATE ORDERS ---
print(f"   Generating {NUM_ORDERS} Orders...")
orders = []
order_ids = [fake.unique.uuid4() for _ in range(NUM_ORDERS)] # Save IDs for items
customer_ids = df_customers['customer_id'].values

for i in range(NUM_ORDERS):
    orders.append({
        'order_id': order_ids[i],
        'customer_id': np.random.choice(customer_ids), # Link to existing customer
        'order_date': fake.date_between(start_date='-1y', end_date='today'),
        'status': np.random.choice(['placed', 'shipped', 'completed', 'returned'], p=[0.1, 0.2, 0.6, 0.1])
    })
df_orders = pd.DataFrame(orders)
# Fix types for Delta
df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
df_orders['customer_id'] = df_orders['customer_id'].astype('int32')

# --- 5. GENERATE ORDER ITEMS (The Join Table) ---
print(f"   Generating Order Items (1-5 items per order)...")
order_items = []
product_ids = df_products['product_id'].values

for order_id in order_ids:
    # Each order has 1 to 5 items
    num_items = np.random.randint(1, 6)
    chosen_products = np.random.choice(product_ids, size=num_items, replace=False)

    for prod_id in chosen_products:
        # Get price from product table
        price = df_products.loc[df_products['product_id'] == prod_id, 'price_cents'].values[0]

        order_items.append({
            'order_id': order_id,
            'product_id': prod_id,
            'quantity': np.random.randint(1, 4),
            'amount_cents': price # Capture price at time of purchase
        })

df_items = pd.DataFrame(order_items)
df_items['product_id'] = df_items['product_id'].astype('int32')
df_items['amount_cents'] = df_items['amount_cents'].astype('int32')

# --- 6. WRITE TO DELTA ---
tables = {
    "raw_customers": df_customers,
    "raw_products": df_products,
    "raw_orders": df_orders,
    "raw_order_items": df_items
}

for name, df in tables.items():
    path = os.path.join(DATA_PATH, name)
    print(f"   💾 Writing {name} to {path}...")

    # Ensure clean slate
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    write_deltalake(path, df, mode="overwrite")

print("\n✅ Data Generation Complete. All 4 tables created.")
