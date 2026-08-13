from feast import Entity, ValueType

customer = Entity(
    name='customer_id',
    value_type=ValueType.INT64,
    description='Borrower identifier used to serve credit-risk features.',
)
