"""Build Delta Lake gold features and export a snapshot for Feast's file store."""

from __future__ import annotations

import logging
import os
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from feast import FeatureStore

from apps.monitoring.metrics import JobMetrics


BRONZE_PATH = Path('/shared/lake/bronze/events')
GOLD_PATH = Path('/shared/lake/gold/user_features')
FEAST_EXPORT_PATH = Path('/shared/lake/exports/feast/user_features.parquet')


def load_repo_module(repo_path: Path, module_name: str):
    """Load Feast definitions from the local feature repository, not the Feast package."""
    module_path = repo_path / f'{module_name}.py'
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        raise ValueError('The bronze Delta table is empty.')

    aggregated = events.groupby('user_id', as_index=False).agg(
        event_value_sum=('event_value', 'sum'),
        event_value_mean=('event_value', 'mean'),
    )
    aggregated['event_value_normalized'] = (
        aggregated['event_value_sum'] / aggregated['event_value_mean']
    )
    # This deterministic target keeps the example model trainable from gold data.
    aggregated['target'] = (
        (2.0 * aggregated['event_value_sum'])
        - (1.5 * aggregated['event_value_normalized'])
        + 0.5
    )
    aggregated['event_time'] = datetime.now(timezone.utc)
    return aggregated


def remove_placeholder_registry(repo_path: Path) -> None:
    """Discard the legacy empty registry placeholder so Feast can create one."""
    registry_path = repo_path.parent / 'registry.db'
    if registry_path.exists() and registry_path.read_bytes().strip() == b'':
        registry_path.unlink()
        logging.info('Removed empty Feast registry placeholder at %s', registry_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('feature_engineering')

    try:
        if not BRONZE_PATH.exists():
            raise FileNotFoundError('No bronze Delta table found; run ingestion first.')

        bronze = DeltaTable(str(BRONZE_PATH))
        feature_df = build_features(bronze.to_pandas())
        GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(GOLD_PATH),
            pa.Table.from_pandas(feature_df, preserve_index=False),
            mode='overwrite',
            schema_mode='overwrite',
        )
        gold_version = DeltaTable(str(GOLD_PATH)).version()

        # Feast's local file offline store does not read Delta transaction logs.
        # Exporting this snapshot makes the boundary explicit while Delta remains
        # the system of record for training and feature engineering.
        FEAST_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(feature_df, preserve_index=False), FEAST_EXPORT_PATH)
        logging.info('Wrote %d gold features to %s (version=%d)', len(feature_df), GOLD_PATH, gold_version)

        repo_path = Path(os.environ.get('FEATURE_STORE_REPO', 'feast/feature_repo'))
        remove_placeholder_registry(repo_path)
        store = FeatureStore(repo_path=str(repo_path))
        user = load_repo_module(repo_path, 'entities').user
        user_feature_view = load_repo_module(repo_path, 'feature_views').user_feature_view
        store.apply([user, user_feature_view])
        store.materialize_incremental(end_date=datetime.now(timezone.utc))
        null_rows = int(feature_df[['event_value_sum', 'event_value_normalized']].isna().any(axis=1).sum())
        metrics.publish(
            success=True,
            rows=len(feature_df),
            custom_metrics={
                'bronze_delta_version': bronze.version(),
                'gold_delta_version': gold_version,
                'feature_null_rows': null_rows,
            },
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
