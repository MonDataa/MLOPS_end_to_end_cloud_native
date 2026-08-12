from datetime import timedelta
from feast import FeatureView, FileSource, Field
from feast.types import Float64
from entities import user

FEATURE_PATH = '/shared/lake/exports/feast/user_features.parquet'

user_features_source = FileSource(
    path=FEATURE_PATH,
    event_timestamp_column='event_time',
)

user_feature_view = FeatureView(
    name='user_features',
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name='event_value_sum', dtype=Float64),
        Field(name='event_value_normalized', dtype=Float64),
        Field(name='event_value_mean', dtype=Float64),
    ],
    online=True,
    source=user_features_source,
)
