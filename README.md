# mlops-shared-volume

## Architecture

This repository orchestrates a local Kubernetes MLOps blueprint on Minikube. The current use case is credit-default risk scoring from repository-local CSV source files, Delta Lake tables, Feast/Redis online features, MLflow model registry, and Prometheus/Grafana monitoring.

The flow is:

1. **Ingestion** loads real CSV source files from `data/raw/` into Delta Lake bronze tables under `/shared/lake/bronze`.
2. **Feature engineering** joins customers, loan applications, credit bureau data, and repayment history into silver and gold Delta tables, then exports `/shared/lake/exports/feast/credit_risk_features.parquet` for Feast.
3. **Feast** applies the `credit_risk_features` feature view and materializes online features into Redis.
4. **Training** uses `scikit-learn` logistic regression to train a credit-default classifier, logs metrics and governance artifacts to MLflow, and registers the production model.
5. **Fairness and explainability** write `/shared/model_reports/fairness_report.json`, `/shared/model_reports/model_card.json`, `/shared/model_reports/feature_defaults.json`, and `/shared/model_reports/permutation_importance.csv`.
6. **Serving** loads the production MLflow model, reads online features from Feast/Redis, returns default probability, risk band, reason codes, and Prometheus metrics.
7. **Monitoring** uses Pushgateway for short-lived jobs, Prometheus for scraping, and a provisioned Grafana `MLOps Overview` dashboard.

## Prerequisites

- Docker and Minikube configured with the Docker driver.
- Helm 3 and kubectl.
- WSL/Linux shell for the Makefile workflow.

## Commands

```sh
make reset-minikube
make build-images
make up
make ingest
make features
make train
make serve
```

Useful checks:

```sh
kubectl -n mlops get pods,job,deploy,svc
kubectl -n mlops logs job/mlops-ingest
kubectl -n mlops logs job/mlops-features
kubectl -n mlops logs job/mlops-training
```

Serving:

```sh
kubectl -n mlops port-forward service/mlops-serving 8080:8080
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{"customer_id":1001}'
curl http://localhost:8080/fairness
curl http://localhost:8080/metrics
```

Monitoring:

```sh
kubectl -n mlops port-forward service/mlops-prometheus 9090:9090
kubectl -n mlops port-forward service/mlops-grafana 3000:3000
kubectl -n mlops port-forward service/mlops-pushgateway 9091:9091
```

Grafana is available at `http://localhost:3000` with `admin` / `admin`.

## Data Model

Raw files:

- `data/raw/customers.csv`
- `data/raw/loan_applications.csv`
- `data/raw/credit_bureau.csv`
- `data/raw/repayment_history.csv`

Lakehouse paths:

- Bronze: `/shared/lake/bronze/{customers,loan_applications,credit_bureau,repayment_history}`
- Silver: `/shared/lake/silver/credit_risk_customer_profile`
- Gold: `/shared/lake/gold/credit_risk_features`
- Feast export: `/shared/lake/exports/feast/credit_risk_features.parquet`

## Governance

The training job excludes sensitive attributes from model features but evaluates group behavior for:

- `gender`
- `age_group`

Logged metrics include ROC AUC, average precision, accuracy, precision, recall, demographic parity difference, and equal opportunity difference. Explainability uses permutation importance over the trained `scikit-learn` pipeline.

## Debug

```sh
minikube status
kubectl -n mlops describe job mlops-features
kubectl -n mlops logs deploy/mlops-serving --tail=100
kubectl -n mlops exec deploy/mlops-serving -- python -c "from deltalake import DeltaTable; print(DeltaTable('/shared/lake/gold/credit_risk_features').version())"
```

Pushgateway metrics:

```sh
kubectl -n mlops exec deploy/mlops-serving -- python -c "import urllib.request; print(urllib.request.urlopen('http://mlops-pushgateway:9091/metrics').read().decode()[:2000])"
```

## Limitations

- The bundled credit-risk dataset is small and repository-local, so it is suitable for an end-to-end MLOps demo, not production model quality.
- Delta Lake is local-PVC backed. For multi-node durable storage, move the table paths to S3-compatible object storage such as MinIO.
- Feast's local file offline store consumes an exported Parquet snapshot; a Delta-aware offline store is the next production migration step.
- Fairness reporting is observational. Real credit decisions need policy review, legal validation, model calibration, and approval workflow.
