import logging
import os
from dataclasses import dataclass
from typing import Tuple

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd

MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///shared/mlruns')
MODEL_NAME = 'mlops-production-model'
FEATURE_COLUMNS = ['event_value_sum', 'event_value_normalized']


def generate_dataset(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = rng.uniform(-1, 1, size=(num_samples, len(FEATURE_COLUMNS))).astype(np.float32)
    weights = np.array([2.0, -1.5], dtype=np.float32)
    bias = 0.5
    noise = rng.normal(scale=0.1, size=num_samples).astype(np.float32)
    targets = (features @ weights) + bias + noise
    return features, targets


def fit_linear_model(features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, float, float]:
    ones = np.ones((features.shape[0], 1), dtype=np.float32)
    design_matrix = np.hstack((ones, features))
    theta = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ targets
    bias = float(theta[0])
    weights = theta[1:]
    predictions = design_matrix @ theta
    mse = float(np.mean((predictions - targets) ** 2))
    return weights, bias, mse


@dataclass
class LinearRegressionModel(mlflow.pyfunc.PythonModel):
    weights: np.ndarray
    bias: float

    def load_context(self, context):
        pass

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        data = model_input[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        return (data @ self.weights) + self.bias


def ensure_registered_model(client: mlflow.tracking.MlflowClient) -> None:
    try:
        client.get_registered_model(MODEL_NAME)
    except Exception:
        try:
            client.create_registered_model(MODEL_NAME)
        except MlflowException:
            logging.warning('Registered model %s already exists', MODEL_NAME)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f'Starting training run with MLflow URI={MLFLOW_URI}', flush=True)

    features, targets = generate_dataset()
    weights, bias, mse_loss = fit_linear_model(features, targets)

    client = mlflow.tracking.MlflowClient()
    ensure_registered_model(client)

    mlflow.set_experiment('mlops-linear')
    with mlflow.start_run(run_name='pseudo-linear-regression'):
        mlflow.log_param('features', FEATURE_COLUMNS)
        mlflow.log_metric('mse_loss', mse_loss)
        ll_model = LinearRegressionModel(weights, bias)
        mlflow.pyfunc.log_model(
            artifact_path='model',
            python_model=ll_model,
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

    print(
        f'Logged linear model with loss={mse_loss} weights={weights.tolist()} bias={bias}',
        flush=True,
    )
    logging.info(
        'Logged linear model with loss=%s weights=%s bias=%s',
        mse_loss,
        weights.tolist(),
        bias,
    )


if __name__ == '__main__':
    main()
