"""Train and register the model from the Delta Lake gold feature table."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pyfunc
from deltalake import DeltaTable
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd

from apps.monitoring.metrics import JobMetrics


MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:////shared/mlflow/mlflow.db')
MLFLOW_ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', 'file:///shared/mlartifacts')
MODEL_NAME = 'mlops-production-model'
GOLD_PATH = Path(os.environ.get('GOLD_FEATURES_PATH', '/shared/lake/gold/user_features'))
FEATURE_COLUMNS = ['event_value_sum', 'event_value_normalized']
TARGET_COLUMN = 'target'


def load_training_data(path: Path = GOLD_PATH) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError('No gold Delta table found; run feature engineering first.')

    table = DeltaTable(str(path))
    dataset = table.to_pandas()
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f'Gold table is missing required columns: {sorted(missing_columns)}')

    clean_dataset = dataset.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if len(clean_dataset) < 2:
        raise ValueError('At least two complete rows are required for training.')
    return clean_dataset, table.version()


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


def ensure_experiment(client: mlflow.tracking.MlflowClient) -> None:
    if client.get_experiment_by_name('mlops-linear') is None:
        client.create_experiment('mlops-linear', artifact_location=MLFLOW_ARTIFACT_ROOT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('training')
    Path('/shared/mlflow').mkdir(parents=True, exist_ok=True)
    Path('/shared/mlartifacts').mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        dataset, gold_version = load_training_data()
        features = dataset[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        targets = dataset[TARGET_COLUMN].to_numpy(dtype=np.float32)
        weights, bias, mse_loss = fit_linear_model(features, targets)

        client = mlflow.tracking.MlflowClient()
        ensure_registered_model(client)
        ensure_experiment(client)
        mlflow.set_experiment('mlops-linear')
        with mlflow.start_run(run_name='delta-gold-linear-regression'):
            mlflow.log_params({
                'features': ','.join(FEATURE_COLUMNS),
                'target': TARGET_COLUMN,
                'delta_table_path': str(GOLD_PATH),
                'delta_table_version': gold_version,
            })
            mlflow.log_metric('mse_loss', mse_loss)
            mlflow.log_metric('training_rows', len(dataset))
            mlflow.pyfunc.log_model(
                artifact_path='model',
                python_model=LinearRegressionModel(weights, bias),
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
            'Registered model version %s from Delta version %d with mse=%s',
            latest_version.version,
            gold_version,
            mse_loss,
        )
        metrics.publish(
            success=True,
            rows=len(dataset),
            custom_metrics={'gold_delta_version': gold_version, 'mse_loss': mse_loss},
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
