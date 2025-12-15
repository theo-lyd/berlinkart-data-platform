WITH raw_products AS (
    SELECT * FROM {{ source('berlinkart', 'raw_products') }}
)

SELECT
    product_id,
    product_name,
    category,

    -- Use the MACRO we created!
    {{ cents_to_euros('price_cents') }} AS price_euro,
    {{ cents_to_euros('cost_cents') }} AS cost_euro,

    -- Calculate Margin (Business Logic)
    {{ cents_to_euros('price_cents - cost_cents') }} AS margin_euro

FROM raw_products
