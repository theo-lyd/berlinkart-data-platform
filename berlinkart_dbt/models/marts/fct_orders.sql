WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
)

SELECT
    order_id,
    customer_id, -- The Foreign Key
    order_date,
    status,
    amount_cents / 100.0 AS total_amount
FROM orders
