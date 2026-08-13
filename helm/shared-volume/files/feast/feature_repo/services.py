from datetime import datetime

from feast import FeatureStore

from entities import customer
from feature_views import credit_risk_feature_view


FEATURE_REPO = 'feast/feature_repo'


def get_feature_store() -> FeatureStore:
    store = FeatureStore(repo_path=FEATURE_REPO)
    store.apply([customer, credit_risk_feature_view])
    return store


def materialize_features():
    store = get_feature_store()
    end_date = datetime.utcnow()
    store.materialize_incremental(end_date=end_date)
    return store


def fetch_online_features(entity_rows):
    store = get_feature_store()
    return store.get_online_features(
        features=[
            'credit_risk_features:bureau_score',
            'credit_risk_features:debt_to_income',
            'credit_risk_features:credit_history_risk_score',
        ],
        entity_rows=entity_rows,
    ).to_dict()
