with orders as (
    select * from {{ ref('stg_orders') }}
),

customers as (
    select
        customer_id,
        -- We arbitrarily pick the 'min' region (alphabetical first)
        -- In a real scenario, you might pick 'most recent' using logic
        min(region) as region
    from orders
    group by customer_id
)

select * from customers