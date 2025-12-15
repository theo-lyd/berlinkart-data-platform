{% snapshot customer_snapshot %}

{{
    config(
      target_database='dev',
      target_schema='snapshots',
      unique_key='customer_id',

      strategy='check',
      check_cols='all',
    )
}}

-- We select strictly from the raw source, not staging!
-- This captures the data exactly as it arrives.
SELECT * FROM {{ source('berlinkart', 'raw_customers') }}

{% endsnapshot %}
