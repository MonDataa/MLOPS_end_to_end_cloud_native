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
import pandas as pd
from feast import FeatureStore
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////shared/mlflow/mlflow.db')
MODEL_NAME = 'mlops-production-model'
SERVER_HOST = '0.0.0.0'
SERVER_PORT = int(os.environ.get('SERVING_PORT', '8080'))
REPORT_ROOT = Path(os.environ.get('MODEL_REPORT_ROOT', '/shared/model_reports'))

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
feature_importance: list[dict[str, object]] = []
fairness_report: dict[str, object] = {}


def find_model_uri(client: mlflow.tracking.MlflowClient) -> str:
    versions = client.get_latest_versions(MODEL_NAME, stages=['Production'])
    if versions:
        return versions[0].source

    runs = list(client.search_runs(order_by=['attributes.start_time DESC'], max_results=1))
    if not runs:
        raise RuntimeError('No MLflow model available')
    return runs[0].info.artifact_uri.rstrip('/') + '/model'


def load_model() -> mlflow.pyfunc.PyFuncModel:
    client = mlflow.tracking.MlflowClient()
    model_uri = find_model_uri(client)
    logging.info('Loading model from %s', model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def load_json(path: Path, default):
    if not path.exists():
        logging.warning('Report file %s not found; using default payload', path)
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def load_reports() -> None:
    global feature_defaults, feature_importance, fairness_report
    feature_defaults = load_json(REPORT_ROOT / 'feature_defaults.json', {})
    fairness_report = load_json(REPORT_ROOT / 'fairness_report.json', {})
    importance_path = REPORT_ROOT / 'permutation_importance.csv'
    if importance_path.exists():
        feature_importance = pd.read_csv(importance_path).to_dict(orient='records')
    else:
        feature_importance = []


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
    if not feature_importance:
        return []

    reasons = []
    row = features.iloc[0].to_dict()
    for item in feature_importance[:limit]:
        feature = str(item['feature'])
        reasons.append(
                {
                    'feature': feature,
                    'importance': float(item['importance_mean']),
                    'value': json_safe_value(row.get(feature)),
                }
            )
    return reasons


def predict(request: dict[str, object]) -> dict[str, object]:
    customer_id = int(request['customer_id'])
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
        return json_response(
            start_response,
            HTTPStatus.OK,
            {'status': 'ok', 'model_loaded': model is not None, 'use_case': 'credit_default_risk'},
        )

    if path == '/fairness' and method == 'GET':
        return json_response(start_response, HTTPStatus.OK, fairness_report)

    if path in {'/predict', '/explain'} and method == 'POST':
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
    model = load_model()
    run_server()
