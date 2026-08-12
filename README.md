# mlops-shared-volume

## Architecture

This repository orchestrates a local Kubernetes (minikube) MLOps blueprint that relies on one shared PVC mounted at `/shared` across ingestion, feature, training, serving, and monitoring workloads. The flow is:

1. **Ingestion** appends events to the Delta Lake bronze table at `/shared/lake/bronze/events`.
2. **Feature engineering** reads bronze, overwrites the Delta Lake gold table at `/shared/lake/gold/user_features`, then exports an explicit Parquet snapshot at `/shared/lake/exports/feast/user_features.parquet` for Feast's file offline store.
3. **Feast** applies the feature definitions, materializes the export into Redis, and exposes online features to serving.
4. **Linear regression training** reads the Delta gold table, logs the Delta table version and metrics, and registers the model in MLflow. MLflow metadata uses SQLite at `/shared/mlflow/mlflow.db`; artifacts are stored at `/shared/mlartifacts`.
5. **Serving (WSGI)** pulls the production MLflow model, resolves features through Feast/Redis, and exposes health, request, error, fallback, and latency metrics.
6. **Monitoring** uses a Pushgateway for short-lived jobs, Prometheus for scraping, and a provisioned Grafana `MLOps Overview` dashboard.
7. **Helm chart** defines the PVC, Kubernetes jobs, deployments, Pushgateway, Prometheus, and Grafana so every component mounts the same volume.

## Prerequisites

- Docker and minikube installed and configured with the Docker driver.
- Helm 3, kubectl, DVC, MLflow, Feast CLI, and Python 3.11+ available locally.

## Step-by-step commands

1. `make up` – starts Minikube, ensures the metrics server, and deploys the shared PVC, Redis, Pushgateway, Prometheus, and Grafana. When you run Helm manually, prefix the chart directory with `./` (for example `helm upgrade --install mlops-shared-volume ./helm/shared-volume`) so Helm treats it as a local chart instead of trying to resolve a repo named `helm`.
2. `make build-images` – builds and loads the training and serving images into Minikube.
3. `make ingest` – appends synthetic events to the bronze Delta table.
4. `make features` – creates the gold Delta table, validates the basic null-rate metric, exports the Feast snapshot, and materializes Redis.
5. `make train` – trains from the gold Delta table and registers the model with its Delta version in MLflow.
6. `make serve` – deploys the WSGI service with `/predict`, `/healthz`, and `/metrics`.

```sh
make build-images
```

7. Expose the local services and validate them:

```sh
kubectl -n mlops port-forward service/mlops-serving 8080:8080
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{"user_id":1}'
kubectl -n mlops port-forward service/mlops-grafana 3000:3000
```

Grafana is available at `http://localhost:3000` with `admin` / `admin`; the `MLOps Overview` dashboard is provisioned automatically. Prometheus can be exposed with `kubectl -n mlops port-forward service/mlops-prometheus 9090:9090`.

8. `make down` – removes the Helm release and stops Minikube.

## Argo CD integration

The repo now uses the recommended App-of-Apps pattern. `argo/application.yaml` points at the `gitops/apps/` directory, and that folder contains a child application (`gitops/apps/mlops-shared-volume/application.yaml`) which deploys `helm/shared-volume`.

1. Install Argo CD via Helm (or your preferred method) into `argocd` namespace and ensure you can reach the API server (`argocd login …`).
2. Update both `argo/application.yaml` and `gitops/apps/mlops-shared-volume/application.yaml` so `repoURL` points to your actual Git repository.
3. Apply the root application into Argo:
   ```sh
   kubectl apply -f argo/application.yaml
   ```
4. Trigger and monitor syncs:
   ```sh
   argocd app sync mlops-root
   argocd app get mlops-root
   argocd app diff mlops-root
   ```
   Argo CD will recursively create the nested `mlops-shared-volume` application, manage the Helm release in `mlops`, and keep it reconciled with Git.
5. After you push new training/serving images, rebuild them (`make build-images`), run your jobs (`make train`/`make serve` or let Argo handle them), then re-sync with `argocd app sync mlops-root`.

The `Makefile` now includes `argo-apply`, `argo-sync`, `argo-get`, and `argo-delete` helpers to control this workflow without manually typing the CLI commands.

## Debug

- Check PVC contents: `minikube ssh -- ls /shared`
- Job logs: `kubectl -n mlops logs job/<name>`
- Helm resources: `helm -n mlops status $(HELM_RELEASE)`.
- Delta table version: use `python -c "from deltalake import DeltaTable; print(DeltaTable('/shared/lake/gold/user_features').version())"` in a workload with the PVC mounted.
- Pipeline metrics: inspect `http://mlops-pushgateway:9091/metrics` from the namespace or query `mlops_job_last_run_success` in Prometheus.
- MLflow UI: use `mlflow ui --backend-store-uri sqlite:////shared/mlflow/mlflow.db --host 0.0.0.0 --port 5000` inside a pod that mounts the PVC.

## Limitations

- Minimal synthetic data and NumPy/pandas model; no real dataset or hyperparameter sweep.
- Delta Lake is local-PVC backed. For multi-node, durable storage, migrate the table paths to S3-compatible object storage such as MinIO.
- Feast's local file offline store still consumes an exported Parquet snapshot; a Delta-aware offline store is the next production migration step.
- No Alertmanager, data-drift detector, backup, or multi-zone redundancy is included.

## Migration path toward a lakehouse

1. Move `/shared/lake` to MinIO/S3 and configure Delta Lake credentials through Kubernetes Secrets.
2. Replace the Feast Parquet export with a Delta-aware offline store or a managed feature platform.
3. Add Alertmanager plus data-quality/drift checks (for example Great Expectations and Evidently) that publish their results to Prometheus and MLflow.
4. Swap the linear regression job for a fuller training script (Spark, Ray, PyTorch, etc.) that consumes from the lakehouse and persists outputs into a catalog.

## Simplified storage option

- The Helm chart exposes `sharedVolume.useHostPath` in `helm/shared-volume/values.yaml`. It now defaults to `false`, so workloads mount the shared PVC from `sharedVolume` instead of a host path. If you need the host-path shortcut while iterating locally, pass overrides to the Makefile (for example `make up HELM_SET="--set sharedVolume.useHostPath=true --set sharedVolume.hostPath=/tmp/mlops-shared"`) and prepare the directory on Minikube with `minikube ssh -- "sudo mkdir -p /tmp/mlops-shared && sudo chown docker:docker /tmp/mlops-shared"`.
