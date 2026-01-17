from datetime import datetime
from feast import FeatureStore
from feast.feature_repo.feature_views import user_feature_view

FEATURE_REPO = 'feast/feature_repo'


def get_feature_store() -> FeatureStore:
    store = FeatureStore(repo_path=FEATURE_REPO)
    store.apply([user_feature_view])
    return store


def materialize_features():
    store = get_feature_store()
    end_date = datetime.utcnow()
    store.materialize_incremental(end_date=end_date)
    return store


def fetch_online_features(entity_df):
    store = get_feature_store()
    return store.get_online_features(
        feature_refs=['user_features:event_value_sum', 'user_features:event_value_normalized'],
        entity_rows=entity_df,
    ).to_dict()
