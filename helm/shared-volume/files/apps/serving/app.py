import json
import logging
import os
from http import HTTPStatus
from typing import Iterable, Optional
from wsgiref.simple_server import make_server

import mlflow
import pandas as pd
from feast import FeatureStore
from prometheus_client import Counter, CONTENT_TYPE_LATEST, generate_latest

MLFLOW_URI = 'file:///shared/mlruns'
MODEL_NAME = 'mlops-production-model'
SERVER_HOST = '0.0.0.0'
SERVER_PORT = int(os.environ.get('SERVING_PORT', '8000'))

REQUESTS = Counter('mlops_predictions_total', 'Total prediction requests')

def _repo_has_config(path: str) -> bool:
    path = os.path.join(path, 'feature_store.yaml')
    return os.path.isfile(path) and os.path.getsize(path) > 0

# Prefer the shared host volume, but use the embedded copy until the shared
# volume has the full repo data.
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


def find_model_uri(client: mlflow.tracking.MlflowClient) -> str:
    versions = client.get_latest_versions(MODEL_NAME, stages=['Production'])
    if versions:
        return versions[0].source

    runs = list(
        client.search_runs(order_by=['attributes.start_time DESC'], max_results=1),
    )
    if not runs:
        raise RuntimeError('No MLflow model available')

    return runs[0].info.artifact_uri.rstrip('/') + '/model'


def load_model() -> mlflow.pyfunc.PyFuncModel:
    client = mlflow.tracking.MlflowClient()
    model_uri = find_model_uri(client)
    logging.info('loading model from %s', model_uri)
    return mlflow.pyfunc.load_model(model_uri)


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


def _default_features() -> pd.DataFrame:
    return pd.DataFrame({
        'event_value_sum': [0.0],
        'event_value_normalized': [0.0],
    })


def predict(request: dict[str, object]) -> dict[str, object]:
    REQUESTS.inc()
    user_id = int(request.get('user_id'))
    try:
        features = store.get_online_features(
            feature_refs=[
                'user_features:event_value_sum',
                'user_features:event_value_normalized',
            ],
            entity_rows=[{'user_id': user_id}],
        ).to_df()
    except Exception:
        logging.exception('Feast could not fetch online features for user %s; falling back to defaults', user_id)
        features = pd.DataFrame()

    if features.empty:
        logging.warning('No online features found for user %s; using defaults', user_id)
        prediction_inputs = _default_features()
    else:
        prediction_inputs = features[['event_value_sum', 'event_value_normalized']]
    values = model.predict(prediction_inputs)
    prediction = float(values[0]) if len(values) else float(values)

    return {'user_id': user_id, 'prediction': prediction}


def application(environ, start_response):
    method = environ.get('REQUEST_METHOD', 'GET')
    path = environ.get('PATH_INFO', '/')

    if path == '/predict' and method == 'POST':
        try:
            length = int(environ.get('CONTENT_LENGTH', '0') or 0)
            raw_body = environ['wsgi.input'].read(length) if length else environ['wsgi.input'].read()
            payload = json.loads(raw_body.decode('utf-8'))
            response = predict(payload)
            return json_response(start_response, HTTPStatus.OK, response)
        except Exception as exc:
            logging.exception('predict request failed')
            return json_response(
                start_response,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {'error': str(exc)},
            )

    if path == '/metrics' and method == 'GET':
        start_response('200 OK', [('Content-Type', CONTENT_TYPE_LATEST)])
        return [generate_latest()]

    start_response('404 Not Found', [('Content-Type', 'text/plain')])
    return [b'not found']


def run_server(host: str = SERVER_HOST, port: int = SERVER_PORT) -> None:
    logging.info('serving on %s:%d', host, port)
    with make_server(host, port, application) as httpd:
        httpd.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model = load_model()
    except Exception as exc:  # keep module importable even if model load fails
        logging.exception('failed to load model during startup: %s', exc)
        raise

    run_server()
