"""WSGI credit-risk prediction service with Prometheus metrics and explanations."""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Iterable, Optional
from wsgiref.simple_server import make_server

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException, RestException
import numpy as np
import pandas as pd
import shap
from feast import FeatureStore
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////shared/mlflow/mlflow.db')
MODEL_NAME = 'mlops-production-model'
EXPERIMENT_NAME = 'credit-risk'
MODEL_STAGE = os.environ.get('MODEL_STAGE', 'Production')
SERVER_HOST = '0.0.0.0'
SERVER_PORT = int(os.environ.get('SERVING_PORT', '8080'))
REPORT_ROOT = Path(os.environ.get('MODEL_REPORT_ROOT', '/shared/model_reports'))
SHAP_BACKGROUND_PATH = REPORT_ROOT / 'shap_background.parquet'
SHAP_SUMMARY_PATH = REPORT_ROOT / 'shap_summary.csv'

FEATURE_COLUMNS = [
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
]

CATEGORICAL_FEATURES = ['employment_status', 'housing_status', 'purpose']

PREDICTION_REQUESTS = Counter('mlops_prediction_requests_total', 'Prediction requests received.')
PREDICTION_ERRORS = Counter('mlops_prediction_errors_total', 'Prediction requests that failed.')
FEAST_FALLBACKS = Counter('mlops_feature_fallbacks_total', 'Predictions using default features.')
PREDICTION_LATENCY = Histogram('mlops_prediction_latency_seconds', 'Prediction request latency.')


def _repo_has_config(path: str) -> bool:
    config_path = os.path.join(path, 'feature_store.yaml')
    return os.path.isfile(config_path) and os.path.getsize(config_path) > 0


_shared_repo = os.environ.get('FEATURE_STORE_REPO', '/shared/feast/feature_repo')
_embedded_repo = os.path.join(os.getcwd(), 'feast', 'feature_repo')
if _repo_has_config(_shared_repo):
    FEATURE_STORE_REPO = _shared_repo
elif _repo_has_config(_embedded_repo):
    FEATURE_STORE_REPO = _embedded_repo
else:
    raise FileNotFoundError('Feast feature store repo not found in /shared or /app/feast')

store = FeatureStore(repo_path=FEATURE_STORE_REPO)
model = None
feature_defaults: dict[str, object] = {}
shap_summary: pd.DataFrame = pd.DataFrame()
fairness_report: dict[str, object] = {}
shap_explainer = None
shap_feature_names: list[str] = []


def find_model_uri(client: mlflow.tracking.MlflowClient) -> str:
    if MODEL_STAGE.lower() in {'latest', 'latest-run', 'candidate'}:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is not None:
            runs = list(
                client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=['attributes.start_time DESC'],
                    max_results=10,
                )
            )
            for run in runs:
                artifacts = client.list_artifacts(run.info.run_id, path='model')
                if artifacts:
                    return run.info.artifact_uri.rstrip('/') + '/model'
            logging.warning(
                'No valid latest-run model artifact found in experiment %s; falling back to Production',
                EXPERIMENT_NAME,
            )

    stage_to_load = 'Production' if MODEL_STAGE.lower() in {'latest', 'latest-run', 'candidate'} else MODEL_STAGE
    versions = []
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage_to_load])
    except (MlflowException, RestException) as exc:
        logging.warning(
            'Registered model %s not found in registry (%s); falling back to latest run',
            MODEL_NAME,
            exc,
        )

    if versions:
        return versions[0].source

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        runs = list(
            client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['attributes.start_time DESC'],
                max_results=1,
            )
        )
    else:
        runs = list(client.search_runs(order_by=['attributes.start_time DESC'], max_results=1))
    if not runs:
        raise RuntimeError('No MLflow model available')
    model_uri = runs[0].info.artifact_uri.rstrip('/') + '/model'
    artifacts = client.list_artifacts(runs[0].info.run_id, path='model')
    if not artifacts:
        raise RuntimeError(f'No model artifact found at {model_uri}')
    return model_uri


def load_model() -> mlflow.pyfunc.PyFuncModel:
    client = mlflow.tracking.MlflowClient()
    model_uri = find_model_uri(client)
    logging.info('Loading model from %s', model_uri)
    return mlflow.sklearn.load_model(model_uri)


def load_json(path: Path, default):
    if not path.exists():
        logging.warning('Report file %s not found; using default payload', path)
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def load_reports() -> None:
    global feature_defaults, shap_summary, fairness_report
    feature_defaults = load_json(REPORT_ROOT / 'feature_defaults.json', {})
    fairness_report = load_json(REPORT_ROOT / 'fairness_report.json', {})
    if SHAP_SUMMARY_PATH.exists():
        shap_summary = pd.read_csv(SHAP_SUMMARY_PATH)
    else:
        shap_summary = pd.DataFrame()


def to_dense_matrix(matrix):
    return matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)


def normalized_shap_feature_name(transformed_name: str) -> str:
    base_name = transformed_name.split('__', 1)[1] if '__' in transformed_name else transformed_name
    for column in CATEGORICAL_FEATURES:
        prefix = f'{column}_'
        if base_name.startswith(prefix):
            return column
    return base_name


def build_shap_explainer() -> None:
    global shap_explainer, shap_feature_names
    if model is None:
        shap_explainer = None
        shap_feature_names = []
        return
    if not SHAP_BACKGROUND_PATH.exists():
        logging.warning('SHAP background file %s not found; local explanations disabled', SHAP_BACKGROUND_PATH)
        shap_explainer = None
        shap_feature_names = []
        return

    background = pd.read_parquet(SHAP_BACKGROUND_PATH)
    if background.empty:
        shap_explainer = None
        shap_feature_names = []
        return

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    shap_feature_names = list(preprocessor.get_feature_names_out())
    background_matrix = to_dense_matrix(preprocessor.transform(background[FEATURE_COLUMNS]))
    shap_explainer = shap.LinearExplainer(classifier, background_matrix)


def json_response(
    start_response,
    status: HTTPStatus,
    payload: object,
    headers: Optional[Iterable[tuple[str, str]]] = None,
) -> list[bytes]:
    body = json.dumps(payload).encode('utf-8')
    response_headers = [('Content-Type', 'application/json')]
    if headers:
        response_headers.extend(headers)
    start_response(f'{status.value} {status.phrase}', response_headers)
    return [body]


def fetch_features(customer_id: int) -> tuple[pd.DataFrame, bool]:
    feature_refs = [f'credit_risk_features:{column}' for column in FEATURE_COLUMNS]
    try:
        online_features = store.get_online_features(
            features=feature_refs,
            entity_rows=[{'customer_id': customer_id}],
        ).to_df()
    except Exception:
        logging.exception('Feast could not fetch online features for customer %s', customer_id)
        online_features = pd.DataFrame()

    if online_features.empty:
        return pd.DataFrame([feature_defaults]).reindex(columns=FEATURE_COLUMNS), True

    features = online_features.reindex(columns=FEATURE_COLUMNS)
    used_defaults = False
    for column in FEATURE_COLUMNS:
        if column not in features or pd.isna(features.at[0, column]):
            features.at[0, column] = feature_defaults.get(column)
            used_defaults = True
    return features, used_defaults


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return 'high'
    if probability >= 0.40:
        return 'medium'
    return 'low'


def json_safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, 'item'):
        return value.item()
    return value


def top_reason_codes(features: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    if shap_explainer is None or not shap_feature_names:
        return []

    row = features[FEATURE_COLUMNS]
    transformed = to_dense_matrix(model.named_steps['preprocessor'].transform(row))
    shap_values = shap_explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)[0]
    original_row = features.iloc[0].to_dict()

    reasons = []
    ranked_indexes = np.argsort(np.abs(values))[::-1][:limit]
    for index in ranked_indexes:
        transformed_name = shap_feature_names[index]
        raw_feature = normalized_shap_feature_name(transformed_name)
        reasons.append(
            {
                'feature': raw_feature,
                'shap_feature': transformed_name,
                'shap_value': float(values[index]),
                'value': json_safe_value(original_row.get(raw_feature)),
            }
        )
    return reasons


def predict(request: dict[str, object]) -> dict[str, object]:
    customer_id_raw = request.get('customer_id', request.get('user_id'))
    if customer_id_raw is None:
        raise KeyError('customer_id')
    customer_id = int(customer_id_raw)
    features, used_defaults = fetch_features(customer_id)
    if used_defaults:
        FEAST_FALLBACKS.inc()
        logging.warning('Incomplete online features for customer %s; defaults were used', customer_id)

    probability = float(model.predict(features)[0])
    return {
        'customer_id': customer_id,
        'default_probability': probability,
        'risk_band': risk_band(probability),
        'used_feature_defaults': used_defaults,
        'top_reason_codes': top_reason_codes(features),
    }


def application(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/')

    if path == '/healthz' and method == 'GET':
        status = HTTPStatus.OK if model is not None else HTTPStatus.SERVICE_UNAVAILABLE
        return json_response(
            start_response,
            status,
            {'status': 'ok' if model is not None else 'degraded', 'model_loaded': model is not None, 'use_case': 'credit_default_risk'},
        )

    if path == '/fairness' and method == 'GET':
        return json_response(start_response, HTTPStatus.OK, fairness_report)

    if path in {'/predict', '/explain'} and method == 'POST':
        if model is None:
            return json_response(
                start_response,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {'error': 'model not loaded yet'},
            )
        PREDICTION_REQUESTS.inc()
        with PREDICTION_LATENCY.time():
            try:
                length = int(environ.get('CONTENT_LENGTH', '0') or 0)
                raw_body = environ['wsgi.input'].read(length) if length else environ['wsgi.input'].read()
                return json_response(start_response, HTTPStatus.OK, predict(json.loads(raw_body.decode('utf-8'))))
            except Exception as exc:
                PREDICTION_ERRORS.inc()
                logging.exception('Prediction request failed')
                return json_response(start_response, HTTPStatus.INTERNAL_SERVER_ERROR, {'error': str(exc)})

    if path == '/metrics' and method == 'GET':
        start_response('200 OK', [('Content-Type', CONTENT_TYPE_LATEST)])
        return [generate_latest()]

    start_response('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'not found']


def run_server(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    logging.info('Serving on %s:%d', host, port)
    with make_server(host, port, application) as httpd:
        httpd.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(MLFLOW_URI)
    load_reports()
    try:
        model = load_model()
        build_shap_explainer()
    except Exception:
        logging.exception('Model not available at startup; serving in degraded mode')
        model = None
    run_server()
