with source as (
    -- Use the variable we defined in dbt_project.yml
    select * from delta_scan('{{ var("raw_orders_path") }}')
),

renamed as (
    select
        order_id,
        customer_id,
        -- Convert string to actual date object
        cast(order_date as date) as order_date,
        region,
        -- Example transformation: Convert EUR to Cents (Integer math is safer)
        cast(amount * 100 as integer) as amount_cents,
        status
    from source
)

select * from renamed