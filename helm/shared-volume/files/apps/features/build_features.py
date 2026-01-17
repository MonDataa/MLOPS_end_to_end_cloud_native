import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from feast import FeatureStore
from feast.feature_repo.feature_views import user_feature_view

RAW_DIR = Path('/shared/data/raw')
FEATURE_DIR = Path('/shared/data/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

csv_files = [p for p in RAW_DIR.glob('*.csv')]
if not csv_files:
    raise FileNotFoundError('No raw CSV files found; run ingestion first.')

frames = [pd.read_csv(p) for p in csv_files]
df = pd.concat(frames, ignore_index=True)

feature_df = (
    df.groupby('user_id', as_index=False)
    .agg(event_value_sum=('event_value', 'sum'))
)
feature_df['event_value_mean'] = df.groupby('user_id')['event_value'].transform('mean')
feature_df['event_value_normalized'] = feature_df['event_value_sum'] / feature_df['event_value_mean']
feature_df['event_time'] = pd.Timestamp(datetime.utcnow())

feature_path = FEATURE_DIR / 'user_features.parquet'
feature_df.to_parquet(feature_path, index=False)
print(f'Features written to {feature_path}')

store = FeatureStore(repo_path='feast/feature_repo')
store.apply([user_feature_view])
now = datetime.utcnow()
start = now - timedelta(hours=24)
store.materialize_incremental(end_date=now)
print('Feast materialization completed')
