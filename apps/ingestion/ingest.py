"""Load raw credit-risk CSV files into Delta Lake bronze tables."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from apps.monitoring.metrics import JobMetrics


RAW_DATA_PATH = Path(os.environ.get('RAW_DATA_PATH', '/shared/data/raw'))
BRONZE_ROOT = Path(os.environ.get('BRONZE_ROOT', '/shared/lake/bronze'))

TABLES = {
    'customers': {
        'file': 'customers.csv',
        'required': ['customer_id', 'birth_year', 'gender', 'state', 'annual_income', 'years_employed'],
        'key': 'customer_id',
    },
    'loan_applications': {
        'file': 'loan_applications.csv',
        'required': [
            'application_id',
            'customer_id',
            'application_date',
            'loan_amount',
            'loan_term_months',
            'interest_rate',
            'employment_status',
            'housing_status',
            'purpose',
            'requested_payment',
            'defaulted',
        ],
        'key': 'application_id',
    },
    'credit_bureau': {
        'file': 'credit_bureau.csv',
        'required': [
            'customer_id',
            'bureau_score',
            'open_accounts',
            'delinquencies_2y',
            'inquiries_6m',
            'revolving_utilization',
            'debt_to_income',
        ],
        'key': 'customer_id',
    },
    'repayment_history': {
        'file': 'repayment_history.csv',
        'required': [
            'customer_id',
            'payments_on_time_12m',
            'payments_late_12m',
            'months_since_last_late',
            'previous_defaults',
        ],
        'key': 'customer_id',
    },
}


def validate_table(name: str, frame: pd.DataFrame, required: list[str], key: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f'{name} is missing required columns: {missing}')
    if frame[key].isna().any():
        raise ValueError(f'{name} contains null values in key column {key}')
    if frame[key].duplicated().any():
        duplicates = frame.loc[frame[key].duplicated(), key].tolist()
        raise ValueError(f'{name} contains duplicate keys in {key}: {duplicates}')


def read_raw_table(name: str, config: dict[str, object]) -> pd.DataFrame:
    path = RAW_DATA_PATH / str(config['file'])
    if not path.exists():
        raise FileNotFoundError(f'Missing raw source file: {path}')

    frame = pd.read_csv(path)
    validate_table(name, frame, list(config['required']), str(config['key']))

    if name == 'loan_applications':
        frame['application_date'] = pd.to_datetime(frame['application_date'], utc=True)
        frame['defaulted'] = frame['defaulted'].astype(int)

    frame['source_file'] = path.name
    frame['ingested_at'] = datetime.now(timezone.utc)
    return frame


def write_bronze_table(name: str, frame: pd.DataFrame) -> int:
    table_path = BRONZE_ROOT / name
    table_path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        str(table_path),
        pa.Table.from_pandas(frame, preserve_index=False),
        mode='overwrite',
        schema_mode='overwrite',
    )
    return DeltaTable(str(table_path)).version()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('ingestion')

    try:
        row_count = 0
        versions: dict[str, int] = {}
        for name, config in TABLES.items():
            frame = read_raw_table(name, config)
            versions[name] = write_bronze_table(name, frame)
            row_count += len(frame)
            logging.info('Loaded %d rows from %s into bronze.%s version %d', len(frame), config['file'], name, versions[name])

        metrics.publish(
            success=True,
            rows=row_count,
            custom_metrics={
                'bronze_table_count': len(TABLES),
                'bronze_max_delta_version': max(versions.values()),
            },
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
