WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
)

SELECT
    order_id,
    customer_id, -- The Foreign Key
    order_date,
    status,
    amount_cents
FROM orders
