WITH snapshot_data AS (
    -- We refer to the snapshot, NOT the raw table
    SELECT * FROM {{ ref('customer_snapshot') }}
)

SELECT
    -- Surrogate Key (Good practice to hash the natural key + uniqueness)
    -- Primary Key for the Dimension (Surrogate Key)
    customer_id,

    -- Attributes
    name,
    email,
    city,
    address,
    signup_date,

    -- SCD Type 2 Metadata (History Tracking)
    -- Snapshot Metadata (Business Logic)
    dbt_valid_from AS valid_from,
    dbt_valid_to AS valid_to,

    -- Boolean flag: "Is this the customer's current address?"
    (dbt_valid_to IS NULL) AS is_current

FROM snapshot_data
