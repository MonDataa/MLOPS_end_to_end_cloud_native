"""Build credit-risk silver and gold Delta tables, then materialize Feast online features."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from feast import FeatureStore

from apps.monitoring.metrics import JobMetrics


BRONZE_ROOT = Path(os.environ.get('BRONZE_ROOT', '/shared/lake/bronze'))
SILVER_PATH = Path(os.environ.get('SILVER_CREDIT_PROFILE_PATH', '/shared/lake/silver/credit_risk_customer_profile'))
GOLD_PATH = Path(os.environ.get('GOLD_FEATURES_PATH', '/shared/lake/gold/credit_risk_features'))
FEAST_EXPORT_PATH = Path(os.environ.get('FEAST_EXPORT_PATH', '/shared/lake/exports/feast/credit_risk_features.parquet'))


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


def read_bronze_table(name: str) -> tuple[pd.DataFrame, int]:
    table_path = BRONZE_ROOT / name
    if not table_path.exists():
        raise FileNotFoundError(f'Missing bronze Delta table {table_path}; run ingestion first.')
    table = DeltaTable(str(table_path))
    return table.to_pandas(), table.version()


def build_credit_profile(
    applications: pd.DataFrame,
    customers: pd.DataFrame,
    bureau: pd.DataFrame,
    repayment: pd.DataFrame,
) -> pd.DataFrame:
    applications = applications[
        [
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
        ]
    ].copy()
    customers = customers[
        ['customer_id', 'birth_year', 'gender', 'state', 'annual_income', 'years_employed']
    ].copy()
    bureau = bureau[
        [
            'customer_id',
            'bureau_score',
            'open_accounts',
            'delinquencies_2y',
            'inquiries_6m',
            'revolving_utilization',
            'debt_to_income',
        ]
    ].copy()
    repayment = repayment[
        [
            'customer_id',
            'payments_on_time_12m',
            'payments_late_12m',
            'months_since_last_late',
            'previous_defaults',
        ]
    ].copy()

    profile = applications.merge(customers, on='customer_id', how='left', validate='one_to_one')
    profile = profile.merge(bureau, on='customer_id', how='left', validate='one_to_one')
    profile = profile.merge(repayment, on='customer_id', how='left', validate='one_to_one')

    critical_columns = ['annual_income', 'bureau_score', 'debt_to_income', 'payments_late_12m']
    if profile[critical_columns].isna().any(axis=None):
        missing_rows = int(profile[critical_columns].isna().any(axis=1).sum())
        raise ValueError(f'{missing_rows} joined rows contain missing critical credit-risk fields')

    event_time = datetime.now(timezone.utc)
    profile['event_timestamp'] = event_time
    profile['application_year'] = pd.to_datetime(profile['application_date'], utc=True).dt.year
    profile['customer_age'] = profile['application_year'] - profile['birth_year']
    profile['age_group'] = pd.cut(
        profile['customer_age'],
        bins=[0, 30, 45, 120],
        labels=['under_30', '30_to_45', 'over_45'],
        right=False,
    ).astype(str)
    profile['monthly_income'] = profile['annual_income'] / 12.0
    profile['loan_to_income'] = profile['loan_amount'] / profile['annual_income']
    profile['installment_to_income'] = profile['requested_payment'] / profile['monthly_income']
    profile['late_payment_rate_12m'] = profile['payments_late_12m'] / (
        profile['payments_late_12m'] + profile['payments_on_time_12m']
    )
    profile['credit_history_risk_score'] = (
        (850 - profile['bureau_score']) / 300
        + profile['delinquencies_2y'] * 0.25
        + profile['previous_defaults'] * 0.50
        + profile['revolving_utilization'] * 0.40
    )
    return profile


def remove_placeholder_registry(repo_path: Path) -> None:
    """Discard the legacy empty registry placeholder so Feast can create one."""
    registry_path = repo_path.parent / 'registry.db'
    if registry_path.exists() and registry_path.read_bytes().strip() == b'':
        registry_path.unlink()
        logging.info('Removed empty Feast registry placeholder at %s', registry_path)


def write_delta(path: Path, frame: pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        str(path),
        pa.Table.from_pandas(frame, preserve_index=False),
        mode='overwrite',
        schema_mode='overwrite',
    )
    return DeltaTable(str(path)).version()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('feature_engineering')

    try:
        applications, applications_version = read_bronze_table('loan_applications')
        customers, customers_version = read_bronze_table('customers')
        bureau, bureau_version = read_bronze_table('credit_bureau')
        repayment, repayment_version = read_bronze_table('repayment_history')

        profile = build_credit_profile(applications, customers, bureau, repayment)
        silver_version = write_delta(SILVER_PATH, profile)

        feature_columns = [
            'customer_id',
            'application_id',
            'event_timestamp',
            'bureau_score',
            'open_accounts',
            'delinquencies_2y',
            'inquiries_6m',
            'revolving_utilization',
            'debt_to_income',
            'annual_income',
            'years_employed',
            'loan_amount',
            'loan_term_months',
            'interest_rate',
            'requested_payment',
            'loan_to_income',
            'installment_to_income',
            'payments_late_12m',
            'late_payment_rate_12m',
            'months_since_last_late',
            'previous_defaults',
            'credit_history_risk_score',
            'employment_status',
            'housing_status',
            'purpose',
            'gender',
            'age_group',
            'defaulted',
        ]
        gold = profile[feature_columns].copy()
        gold_version = write_delta(GOLD_PATH, gold)

        FEAST_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(gold.drop(columns=['defaulted']), preserve_index=False), FEAST_EXPORT_PATH)

        repo_path = Path(os.environ.get('FEATURE_STORE_REPO', 'feast/feature_repo'))
        remove_placeholder_registry(repo_path)
        store = FeatureStore(repo_path=str(repo_path))
        customer = load_repo_module(repo_path, 'entities').customer
        credit_risk_feature_view = load_repo_module(repo_path, 'feature_views').credit_risk_feature_view
        store.apply([customer, credit_risk_feature_view])
        store.materialize_incremental(end_date=datetime.now(timezone.utc))

        null_rows = int(gold.drop(columns=['gender', 'age_group']).isna().any(axis=1).sum())
        logging.info(
            'Wrote %d credit-risk gold feature rows to %s (version=%d)',
            len(gold),
            GOLD_PATH,
            gold_version,
        )
        metrics.publish(
            success=True,
            rows=len(gold),
            custom_metrics={
                'bronze_max_delta_version': max(applications_version, customers_version, bureau_version, repayment_version),
                'silver_delta_version': silver_version,
                'gold_delta_version': gold_version,
                'feature_null_rows': null_rows,
                'default_rate': float(gold['defaulted'].mean()),
            },
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
