WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

order_items AS (
    SELECT * FROM {{ source('berlinkart', 'raw_order_items') }}
    -- Note: In a real project, you'd likely have 'stg_order_items',
    -- but reading raw is fine for this portfolio step.
),

order_aggregates AS (
    SELECT
        order_id,
        COUNT(product_id) AS num_items,
        SUM(amount_cents) AS total_revenue_cents
    FROM order_items
    GROUP BY order_id
)

SELECT
    o.order_id,
    o.customer_id, -- Foreign Key to dim_customers
    o.order_date,
    o.status,

    -- Metrics
    COALESCE(agg.num_items, 0) AS quantity,

    -- Use the Macro for consistent money handling
    {{ cents_to_euros('COALESCE(agg.total_revenue_cents, 0)') }} AS revenue_euro

FROM orders AS o
LEFT JOIN order_aggregates AS agg ON o.order_id = agg.order_id
