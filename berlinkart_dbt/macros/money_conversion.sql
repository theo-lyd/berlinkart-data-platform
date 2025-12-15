{% macro cents_to_euros(column_name, precision=2) %}
    ( {{ column_name }} / 100.0 )::decimal(18, {{ precision }})
{% endmacro %}
