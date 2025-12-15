with orders as (
    select * from {{ ref('stg_orders') }}
)

select
    order_id,
    customer_id, -- The Foreign Key
    order_date,
    status,
    amount_cents
from orders