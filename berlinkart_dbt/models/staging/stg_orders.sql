WITH source AS (
    -- Use the variable we defined in dbt_project.yml
    SELECT * FROM delta_scan('{{ var("raw_orders_path") }}')
),

renamed AS (
    SELECT
        order_id,
        customer_id,
        -- Convert string to actual date object
        cast(order_date AS date) AS order_date,
        region,
        -- Example transformation: Convert EUR to Cents (Integer math is safer)
        cast(amount * 100 AS integer) AS amount_cents,
        status
    FROM source
)

SELECT * FROM renamed
