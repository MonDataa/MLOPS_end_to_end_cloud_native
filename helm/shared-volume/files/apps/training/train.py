"""Train and register a credit-default classifier from Delta Lake gold features."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from deltalake import DeltaTable
from mlflow.exceptions import MlflowException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apps.monitoring.metrics import JobMetrics


MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////shared/mlflow/mlflow.db')
MLFLOW_ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', 'file:///shared/mlartifacts')
MODEL_NAME = 'mlops-production-model'
GOLD_PATH = Path(os.environ.get('GOLD_FEATURES_PATH', '/shared/lake/gold/credit_risk_features'))
REPORT_ROOT = Path(os.environ.get('MODEL_REPORT_ROOT', '/shared/model_reports'))
REFERENCE_FEATURES_PATH = REPORT_ROOT / 'reference_features.parquet'
SHAP_BACKGROUND_PATH = REPORT_ROOT / 'shap_background.parquet'
SHAP_SUMMARY_PATH = REPORT_ROOT / 'shap_summary.csv'
TARGET_COLUMN = 'defaulted'
SENSITIVE_COLUMNS = ['gender', 'age_group']
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


def load_training_data(path: Path = GOLD_PATH) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError('No credit-risk gold Delta table found; run feature engineering first.')

    table = DeltaTable(str(path))
    dataset = table.to_pandas()
    required_columns = set(FEATURE_COLUMNS + SENSITIVE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f'Gold table is missing required columns: {missing_columns}')

    dataset = dataset.dropna(subset=FEATURE_COLUMNS + SENSITIVE_COLUMNS + [TARGET_COLUMN]).copy()
    if dataset[TARGET_COLUMN].nunique() < 2:
        raise ValueError('Training target must contain both classes.')
    return dataset, table.version()


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])


def to_dense_matrix(matrix):
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)


def normalized_shap_feature_name(transformed_name: str) -> str:
    base_name = transformed_name.split('__', 1)[1] if '__' in transformed_name else transformed_name
    for column in CATEGORICAL_FEATURES:
        prefix = f'{column}_'
        if base_name.startswith(prefix):
            return column
    return base_name


def score_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, probabilities)),
        'average_precision': float(average_precision_score(y_test, probabilities)),
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions, zero_division=0)),
        'recall': float(recall_score(y_test, predictions, zero_division=0)),
    }
    return metrics, probabilities, predictions


def build_shap_reports(model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
    background = x_train[FEATURE_COLUMNS].sample(n=min(20, len(x_train)), random_state=42).copy()
    background.to_parquet(SHAP_BACKGROUND_PATH, index=False)

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    feature_names = list(preprocessor.get_feature_names_out())

    background_matrix = to_dense_matrix(preprocessor.transform(background))
    test_matrix = to_dense_matrix(preprocessor.transform(x_test[FEATURE_COLUMNS]))
    explainer = shap.LinearExplainer(classifier, background_matrix)
    shap_values = explainer.shap_values(test_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    summary = pd.DataFrame(
        {
            'feature_name': feature_names,
            'raw_feature': [normalized_shap_feature_name(name) for name in feature_names],
            'mean_abs_shap': np.abs(shap_values).mean(axis=0),
            'mean_shap': shap_values.mean(axis=0),
        }
    ).sort_values('mean_abs_shap', ascending=False)
    summary['mean_abs_shap_pct'] = summary['mean_abs_shap'] / summary['mean_abs_shap'].sum()
    SHAP_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SHAP_SUMMARY_PATH, index=False)
    return summary


def save_reference_features(x_train: pd.DataFrame) -> None:
    REFERENCE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    x_train[FEATURE_COLUMNS].copy().to_parquet(REFERENCE_FEATURES_PATH, index=False)


def group_rates(frame: pd.DataFrame, group_column: str, predictions: np.ndarray) -> dict[str, dict[str, float]]:
    scored = frame[[group_column, TARGET_COLUMN]].copy()
    scored['prediction'] = predictions
    rates: dict[str, dict[str, float]] = {}
    for group, group_frame in scored.groupby(group_column):
        positives = group_frame[group_frame[TARGET_COLUMN] == 1]
        negatives = group_frame[group_frame[TARGET_COLUMN] == 0]
        rates[str(group)] = {
            'rows': float(len(group_frame)),
            'predicted_default_rate': float(group_frame['prediction'].mean()),
            'true_default_rate': float(group_frame[TARGET_COLUMN].mean()),
            'true_positive_rate': float(positives['prediction'].mean()) if len(positives) else 0.0,
            'false_positive_rate': float(negatives['prediction'].mean()) if len(negatives) else 0.0,
        }
    return rates


def max_difference(rates: dict[str, dict[str, float]], metric_name: str) -> float:
    values = [group_metrics[metric_name] for group_metrics in rates.values()]
    return float(max(values) - min(values)) if values else 0.0


def build_fairness_report(test_frame: pd.DataFrame, predictions: np.ndarray) -> tuple[dict[str, object], dict[str, float]]:
    report: dict[str, object] = {}
    metrics: dict[str, float] = {}
    for column in SENSITIVE_COLUMNS:
        rates = group_rates(test_frame, column, predictions)
        report[column] = rates
        metrics[f'demographic_parity_{column}_diff'] = max_difference(rates, 'predicted_default_rate')
        metrics[f'equal_opportunity_{column}_diff'] = max_difference(rates, 'true_positive_rate')
    return report, metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def write_explainability_report(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    del model
    del x_test
    del y_test
    if not SHAP_SUMMARY_PATH.exists():
        raise FileNotFoundError(f'Missing SHAP summary at {SHAP_SUMMARY_PATH}')
    return pd.read_csv(SHAP_SUMMARY_PATH)


def feature_defaults(dataset: pd.DataFrame) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for column in NUMERIC_FEATURES:
        defaults[column] = float(dataset[column].median())
    for column in CATEGORICAL_FEATURES:
        defaults[column] = str(dataset[column].mode(dropna=True).iloc[0])
    return defaults


def ensure_registered_model(client: mlflow.tracking.MlflowClient) -> None:
    try:
        client.get_registered_model(MODEL_NAME)
    except Exception:
        try:
            client.create_registered_model(MODEL_NAME)
        except MlflowException:
            logging.warning('Registered model %s already exists', MODEL_NAME)


def ensure_experiment(client: mlflow.tracking.MlflowClient) -> None:
    if client.get_experiment_by_name('credit-risk') is None:
        client.create_experiment('credit-risk', artifact_location=MLFLOW_ARTIFACT_ROOT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('training')
    Path('/shared/mlflow').mkdir(parents=True, exist_ok=True)
    Path('/shared/mlartifacts').mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        dataset, gold_version = load_training_data()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[FEATURE_COLUMNS + SENSITIVE_COLUMNS],
            dataset[TARGET_COLUMN].astype(int),
            test_size=0.30,
            random_state=42,
            stratify=dataset[TARGET_COLUMN],
        )

        model = build_pipeline()
        model.fit(x_train[FEATURE_COLUMNS], y_train)
        model_metrics, probabilities, predictions = score_model(model, x_test[FEATURE_COLUMNS], y_test)
        test_frame = x_test.copy()
        test_frame[TARGET_COLUMN] = y_test.to_numpy()
        fairness_report, fairness_metrics = build_fairness_report(test_frame, predictions)
        save_reference_features(x_train)
        shap_summary = build_shap_reports(model, x_train, x_test)

        report_payload = {
            'use_case': 'credit_default_risk',
            'target': TARGET_COLUMN,
            'sensitive_attributes': SENSITIVE_COLUMNS,
            'model_features': FEATURE_COLUMNS,
            'metrics': model_metrics,
            'fairness': fairness_report,
            'shap_top_feature': str(shap_summary.iloc[0]['raw_feature']),
            'delta_gold_version': gold_version,
            'train_rows': int(len(x_train)),
            'test_rows': int(len(x_test)),
        }
        write_json(REPORT_ROOT / 'fairness_report.json', fairness_report)
        write_json(REPORT_ROOT / 'model_card.json', report_payload)
        write_json(REPORT_ROOT / 'feature_defaults.json', feature_defaults(dataset))

        client = mlflow.tracking.MlflowClient()
        ensure_registered_model(client)
        ensure_experiment(client)
        mlflow.set_experiment('credit-risk')
        with mlflow.start_run(run_name='credit-default-logistic-regression'):
            mlflow.log_params(
                {
                    'model_type': 'sklearn.LogisticRegression',
                    'features': ','.join(FEATURE_COLUMNS),
                    'excluded_sensitive_features': ','.join(SENSITIVE_COLUMNS),
                    'target': TARGET_COLUMN,
                    'delta_table_path': str(GOLD_PATH),
                    'delta_table_version': gold_version,
                }
            )
            mlflow.log_metrics(model_metrics | fairness_metrics)
            mlflow.log_metric('training_rows', len(x_train))
            mlflow.log_metric('test_rows', len(x_test))
            mlflow.log_artifact(str(REPORT_ROOT / 'fairness_report.json'), artifact_path='governance')
            mlflow.log_artifact(str(REPORT_ROOT / 'model_card.json'), artifact_path='governance')
            mlflow.log_artifact(str(REFERENCE_FEATURES_PATH), artifact_path='monitoring')
            mlflow.log_artifact(str(SHAP_BACKGROUND_PATH), artifact_path='explainability')
            mlflow.log_artifact(str(SHAP_SUMMARY_PATH), artifact_path='explainability')
            mlflow.sklearn.log_model(
                sk_model=model,
                name='model',
                serialization_format='cloudpickle',
                registered_model_name=MODEL_NAME,
            )

        versions = client.get_latest_versions(MODEL_NAME)
        if not versions:
            raise RuntimeError(f'Failed to log model {MODEL_NAME}')
        latest_version = max(versions, key=lambda version: int(version.version))
        client.transition_model_version_stage(
            name=latest_version.name,
            version=latest_version.version,
            stage='Production',
            archive_existing_versions=True,
        )

        logging.info(
            'Registered credit-risk model version %s from Delta version %d with roc_auc=%s and top_feature=%s',
            latest_version.version,
            gold_version,
            model_metrics['roc_auc'],
            shap_summary.iloc[0]['raw_feature'],
        )
        metrics.publish(
            success=True,
            rows=len(dataset),
            custom_metrics={
                'gold_delta_version': gold_version,
                'roc_auc': model_metrics['roc_auc'],
                'average_precision': model_metrics['average_precision'],
                'demographic_parity_gender_diff': fairness_metrics['demographic_parity_gender_diff'],
                'equal_opportunity_gender_diff': fairness_metrics['equal_opportunity_gender_diff'],
            },
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
