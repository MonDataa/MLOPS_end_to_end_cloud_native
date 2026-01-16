from feast import Entity, ValueType

user = Entity(
    name='user_id',
    value_type=ValueType.INT64,
    description='Identifier of the user for feature aggregation',
)
