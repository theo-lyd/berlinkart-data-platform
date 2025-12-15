WITH source AS (
    -- Now we use the source configuration from sources.yml
    -- This picks up the absolute path we fixed earlier.
    SELECT * FROM {{ source('berlinkart', 'raw_orders') }}
),

renamed AS (
    SELECT
        order_id,
        customer_id,
        status,

        -- Ensure it's a date type
        CAST(order_date AS date) AS order_date

        -- NOTE: We removed 'amount' and 'region'.
        -- 'amount' is now calculated in fct_orders (from items).
        -- 'region/city' is now joined in from dim_customers.
    FROM source
)

SELECT * FROM renamed
