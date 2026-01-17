# mlops-shared-volume

## Architecture

This repository orchestrates a local Kubernetes (minikube) MLOps blueprint that relies on one shared PVC mounted at `/shared` across ingestion, feature, training, serving, and monitoring workloads. The flow is:

1. **Ingestion** writes raw CSVs to `/shared/data/raw` and versioning is enforced via DVC.
2. **Feature engineering** consumes raw files and emits featurized Parquet files under `/shared/data/features`.
3. **Feast** ingests features from the filesystem for offline access and mirrors them into Redis as the online store.
4. **Linear regression training** loads the offline features via Feast, derives a NumPy-based model, logs experiments in MLflow under `/shared/mlruns`, and registers the artifacts.
5. **Serving (WSGI)** pulls the latest production MLflow model, resolves feature values via Feast online store, and exposes Prometheus counters directly from the lightweight WSGI handler.
6. **Helm chart** defines the PVC, Kubernetes jobs, and deployments so every component mounts the same volume.

## Prerequisites

- Docker and minikube installed and configured with the Docker driver.
- Helm 3, kubectl, DVC, MLflow, Feast CLI, and Python 3.11+ available locally.

## Step-by-step commands

1. `make up` – starts minikube, ensures the metrics server, and deploys the Helm chart that creates the shared PVC, Redis, and placeholders for the workloads. When you run Helm manually, prefix the chart directory with `./` (for example `helm upgrade --install mlops-shared-volume ./helm/shared-volume`) so Helm treats it as a local chart instead of trying to resolve a repo named `helm`.
2. `make ingest` – runs the ingestion job that generates synthetic CSVs into `/shared/data/raw`.
3. `make features` – executes the feature job to materialize `/shared/data/features` and registers them with Feast.
4. `make train` – schedules the training job that generates its own NumPy/pandas dataset, fits the linear regression locally, logs metrics to MLflow, and writes the serialized `mlflow` model under `/shared/mlruns` on the PVC. Rebuild the training container after editing the code or requirements so Minikube sees the latest version (the job sets `imagePullPolicy: IfNotPresent`):

```sh
docker build -t mlops-training:latest apps/training
minikube image load mlops-training:latest
```

Build the serving image once so the lightweight WSGI server does not need to reinstall dependencies on each restart:

```sh
docker build -t mlops-serving:latest apps/serving
minikube image load mlops-serving:latest
```

5. `make serve` – deploys the serving job that installs the minimal requirements, runs `apps/serving/app.py`, and exposes `/predict` plus `/metrics` from the WSGI app.
   * Optional: `make ingest` and `make features` still exist to show how raw data feeds Feast/Redis, but the trainer no longer depends on them to produce a model (they can be re-run if you want real Feast materializations).
6. `curl localhost:8000/predict` – hit the serving endpoint to validate the entire DAG.
7. `make down` – removes the Helm release and stops minikube.

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
- Feast materialization: use `feast materialize` against the files in `/shared/data/features`.
- MLflow UI: `mlflow ui --backend-store-uri /shared/mlruns --host 0.0.0.0 --port 5000` inside a pod that mounts the PVC.

## Limitations

- Minimal synthetic data and NumPy/pandas model; no real dataset or hyperparameter sweep.
- Training job is a single linear-regression run; no autoscaling or distributed compute.
- No advanced monitoring besides Prometheus counters exposed through the WSGI endpoint.
- No backup or multi-zone redundancy for the shared PVC.

## Migration path toward a lakehouse

1. Replace `/shared/data/raw` and `/shared/data/features` with a Delta/Parquet lakehouse, introducing storage like MinIO or local S3-compatible service.
2. Point Feast offline store to the lakehouse tables and use Spark/Arrow connectors for ingestion.
3. Swap the linear regression job for a fuller training script (Spark, Ray, PyTorch, etc.) that consumes from the lakehouse and persists outputs into a catalog.
4. Introduce a metadata layer (e.g., Amundsen/Atlas) backed by the lakehouse for lineage and governance.

## Simplified storage option

- The Helm chart exposes `sharedVolume.useHostPath` in `helm/shared-volume/values.yaml`. It now defaults to `false`, so workloads mount the shared PVC from `sharedVolume` instead of a host path. If you need the host-path shortcut while iterating locally, pass overrides to the Makefile (for example `make up HELM_SET="--set sharedVolume.useHostPath=true --set sharedVolume.hostPath=/tmp/mlops-shared"`) and prepare the directory on Minikube with `minikube ssh -- "sudo mkdir -p /tmp/mlops-shared && sudo chown docker:docker /tmp/mlops-shared"`.
