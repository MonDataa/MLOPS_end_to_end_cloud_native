from datetime import timedelta
from feast import FeatureView, FileSource, Feature, ValueType
from feast.feature_repo.entities import user

FEATURE_PATH = '/shared/data/features/user_features.parquet'

user_features_source = FileSource(
    path=FEATURE_PATH,
    event_timestamp_column='event_time',
    created_timestamp_column='event_time',
)

user_feature_view = FeatureView(
    name='user_features',
    entities=['user_id'],
    ttl=timedelta(days=1),
    schema=[
        Feature(name='event_value_sum', dtype=ValueType.DOUBLE),
        Feature(name='event_value_normalized', dtype=ValueType.DOUBLE),
        Feature(name='event_value_mean', dtype=ValueType.DOUBLE),
    ],
    online=True,
    batch_source=user_features_source,
)
