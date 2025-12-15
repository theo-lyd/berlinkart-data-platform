WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

customers AS (
    SELECT
        customer_id,
        -- We arbitrarily pick the 'min' region (alphabetical first)
        -- In a real scenario, you might pick 'most recent' using logic
        min(region) AS region
    FROM orders
    GROUP BY customer_id
)

SELECT * FROM customers
