"""Drift detection and retraining trigger for the credit-risk pipeline."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from sklearn.metrics import accuracy_score, roc_auc_score

from apps.monitoring.metrics import JobMetrics


MODEL_NAME = 'mlops-production-model'
MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////shared/mlflow/mlflow.db')
REPORT_ROOT = Path(os.environ.get('MODEL_REPORT_ROOT', '/shared/model_reports'))
REFERENCE_FEATURES_PATH = REPORT_ROOT / 'reference_features.parquet'
DRIFT_REPORT_PATH = REPORT_ROOT / 'drift_report.json'
MODEL_CARD_PATH = REPORT_ROOT / 'model_card.json'
CURRENT_GOLD_PATH = Path(os.environ.get('GOLD_FEATURES_PATH', '/shared/lake/gold/credit_risk_features'))
DRIFT_PSI_THRESHOLD = float(os.environ.get('DRIFT_PSI_THRESHOLD', '0.2'))
AUTO_RETRAIN_ON_DRIFT = os.environ.get('AUTO_RETRAIN_ON_DRIFT', 'false').lower() in {'1', 'true', 'yes', 'on'}
SAMPLE_SIZE = int(os.environ.get('DRIFT_SAMPLE_SIZE', '500'))
TARGET_COLUMN = 'defaulted'
NUMERIC_FEATURES = [
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
]
CATEGORICAL_FEATURES = ['employment_status', 'housing_status', 'purpose']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def sanitize_metric_name(value: str) -> str:
    return ''.join(character if character.isalnum() else '_' for character in value)


def load_json(path: Path, default: dict[str, object]) -> dict[str, object]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def load_reference_features() -> pd.DataFrame:
    if not REFERENCE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f'Missing training reference features at {REFERENCE_FEATURES_PATH}; rerun training first.'
        )
    return pd.read_parquet(REFERENCE_FEATURES_PATH)


def load_current_features(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    if not CURRENT_GOLD_PATH.exists():
        raise FileNotFoundError(f'Missing current Delta table at {CURRENT_GOLD_PATH}')
    table = DeltaTable(str(CURRENT_GOLD_PATH))
    current = table.to_pandas()
    if current.empty:
        raise ValueError('Current feature table is empty')
    current = current.tail(sample_size).copy()
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in current.columns]
    if missing_columns:
        raise ValueError(f'Current feature table is missing required columns: {missing_columns}')
    return current


def aligned_non_null(reference: pd.Series, current: pd.Series) -> tuple[pd.Series, pd.Series]:
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return ref, cur
    return ref, cur


def psi_for_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = aligned_non_null(reference, current)
    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.unique(np.quantile(ref.astype(float), np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return psi_for_categorical(ref.astype(str), cur.astype(str))

    ref_bins = pd.cut(ref.astype(float), bins=quantiles, include_lowest=True, duplicates='drop')
    cur_bins = pd.cut(cur.astype(float), bins=quantiles, include_lowest=True, duplicates='drop')
    categories = sorted(set(ref_bins.astype(str).unique()) | set(cur_bins.astype(str).unique()))
    return distribution_psi(ref_bins.astype(str), cur_bins.astype(str), categories)


def psi_for_categorical(reference: pd.Series, current: pd.Series) -> float:
    ref, cur = aligned_non_null(reference.astype(str), current.astype(str))
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if not categories:
        return 0.0
    return distribution_psi(ref, cur, categories)


def distribution_psi(reference: pd.Series, current: pd.Series, categories: list[str]) -> float:
    epsilon = 1e-6
    ref_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0.0) + epsilon
    cur_share = current.value_counts(normalize=True).reindex(categories, fill_value=0.0) + epsilon
    return float(((cur_share - ref_share) * np.log(cur_share / ref_share)).sum())


def evaluate_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict[str, object], dict[str, float]]:
    feature_report: dict[str, object] = {}
    metrics: dict[str, float] = {}
    for feature in FEATURE_COLUMNS:
        reference_series = reference[feature]
        current_series = current[feature]
        if feature in NUMERIC_FEATURES:
            psi = psi_for_numeric(reference_series, current_series)
        else:
            psi = psi_for_categorical(reference_series.astype(str), current_series.astype(str))
        missing_delta = float(current_series.isna().mean() - reference_series.isna().mean())
        feature_report[feature] = {
            'psi': psi,
            'missing_rate_reference': float(reference_series.isna().mean()),
            'missing_rate_current': float(current_series.isna().mean()),
            'missing_rate_delta': missing_delta,
            'drifted': psi >= DRIFT_PSI_THRESHOLD,
        }
        metrics[f'drift_psi_{sanitize_metric_name(feature)}'] = psi
        metrics[f'drift_missing_delta_{sanitize_metric_name(feature)}'] = missing_delta

    max_psi = max((feature_payload['psi'] for feature_payload in feature_report.values()), default=0.0)
    drifted_features = sum(1 for feature_payload in feature_report.values() if feature_payload['drifted'])
    metrics['drift_max_psi'] = float(max_psi)
    metrics['drift_drifted_features_count'] = float(drifted_features)
    return feature_report, metrics


def load_model() -> mlflow.pyfunc.PyFuncModel | None:
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=['Production'])
        if not versions:
            return None
        model_uri = versions[0].source
        logging.info('Loading production model from %s', model_uri)
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        logging.exception('Could not load production model for drift scoring')
        return None


def load_baseline_metrics() -> dict[str, object]:
    return load_json(MODEL_CARD_PATH, {}).get('metrics', {})


def evaluate_concept_drift(model, current: pd.DataFrame) -> dict[str, float]:
    if model is None or TARGET_COLUMN not in current.columns:
        return {}

    scored = current.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if scored.empty or scored[TARGET_COLUMN].nunique() < 2:
        return {}

    probabilities = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        'current_roc_auc': float(roc_auc_score(scored[TARGET_COLUMN].astype(int), probabilities)),
        'current_accuracy': float(accuracy_score(scored[TARGET_COLUMN].astype(int), predictions)),
        'current_rows': float(len(scored)),
    }


def maybe_retrain(needs_retrain: bool) -> bool:
    if not needs_retrain:
        return False

    if not AUTO_RETRAIN_ON_DRIFT:
        logging.warning('Drift detected but automatic retraining is disabled')
        return False

    logging.info('Drift threshold exceeded; launching retraining job')
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', '/shared')
    try:
        subprocess.run([sys.executable, '/shared/apps/training/train.py'], check=True, env=env)
    except subprocess.CalledProcessError:
        logging.exception('Retraining job failed; continuing drift monitoring without retrain')
        return False
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('drift_monitor')
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        reference = load_reference_features()
        current = load_current_features()
        feature_report, feature_metrics = evaluate_feature_drift(reference, current)
        model = load_model()
        baseline_metrics = load_baseline_metrics()
        concept_metrics = evaluate_concept_drift(model, current)

        baseline_auc = float(baseline_metrics.get('roc_auc', 0.0) or 0.0)
        current_auc = float(concept_metrics.get('current_roc_auc', 0.0) or 0.0)
        auc_drop = baseline_auc - current_auc if current_auc and baseline_auc else 0.0

        needs_retrain = feature_metrics['drift_max_psi'] >= DRIFT_PSI_THRESHOLD or auc_drop >= 0.05
        retrained = maybe_retrain(needs_retrain)

        report_payload = {
            'drift_threshold': DRIFT_PSI_THRESHOLD,
            'auto_retrain_on_drift': AUTO_RETRAIN_ON_DRIFT,
            'needs_retrain': needs_retrain,
            'retrained': retrained,
            'reference_rows': int(len(reference)),
            'current_rows': int(len(current)),
            'feature_drift': feature_report,
            'concept_drift': {
                **concept_metrics,
                'baseline_roc_auc': baseline_auc,
                'roc_auc_drop_vs_baseline': auc_drop,
            },
        }
        DRIFT_REPORT_PATH.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding='utf-8')

        custom_metrics = feature_metrics | {
            'drift_needs_retrain': 1.0 if needs_retrain else 0.0,
            'drift_retrained': 1.0 if retrained else 0.0,
            'drift_baseline_roc_auc': baseline_auc,
            'drift_current_roc_auc': current_auc,
            'drift_roc_auc_drop_vs_baseline': auc_drop,
            'drift_reference_rows': float(len(reference)),
            'drift_current_rows': float(len(current)),
        }
        custom_metrics.update(concept_metrics)
        metrics.publish(success=True, rows=len(current), custom_metrics=custom_metrics)
        logging.info(
            'Drift check completed max_psi=%.4f needs_retrain=%s current_auc=%s',
            feature_metrics['drift_max_psi'],
            needs_retrain,
            concept_metrics.get('current_roc_auc'),
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
