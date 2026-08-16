IMAGE_OWNER?=mondataa

HELM_RELEASE=mlops-shared-volume
CHART_DIR=helm/shared-volume
SHARED_PATH=/tmp/mlops-shared
HELM_SET=--set sharedVolume.useHostPath=true --set sharedVolume.hostPath=$(SHARED_PATH)
MINIKUBE_MEMORY?=6000mb
MINIKUBE_CPUS?=2

ARGO_APP=argo/application.yaml
ARGO_NAMESPACE=argocd
ARGO_APP_NAME=mlops-shared-volume

.PHONY: up sync-config ingest features train serve drift pipeline-cron canary down images build-images reset-minikube
.PHONY: drift-manual argo-check argo-apply argo-sync argo-get argo-delete

up:
	minikube start --driver=docker --addons metrics-server --memory=$(MINIKUBE_MEMORY) --cpus=$(MINIKUBE_CPUS)
	minikube ssh -- "sudo mkdir -p $(SHARED_PATH) && sudo chown docker:docker $(SHARED_PATH)"
	$(MAKE) images
	kubectl -n mlops delete job mlops-ingest mlops-features mlops-training --ignore-not-found
	kubectl -n mlops delete cronjob mlops-drift-monitor mlops-pipeline-cron --ignore-not-found
	kubectl -n mlops delete deployment mlops-serving mlops-serving-canary mlops-redis mlops-prometheus mlops-pushgateway mlops-grafana --ignore-not-found
	kubectl -n mlops delete service mlops-serving mlops-serving-canary mlops-redis mlops-prometheus mlops-pushgateway mlops-grafana --ignore-not-found
	helm upgrade --install $(HELM_RELEASE) $(CHART_DIR) --create-namespace --namespace mlops $(HELM_SET)

images: build-images

build-images:
	docker build --no-cache -t mlops-training:latest -t ghcr.io/$(IMAGE_OWNER)/mlops-training:latest -f apps/training/Dockerfile .
	minikube image load mlops-training:latest
	minikube image load ghcr.io/$(IMAGE_OWNER)/mlops-training:latest
	docker build --no-cache -t mlops-serving:latest -t ghcr.io/$(IMAGE_OWNER)/mlops-serving:latest -f apps/serving/Dockerfile .
	minikube image load mlops-serving:latest
	minikube image load ghcr.io/$(IMAGE_OWNER)/mlops-serving:latest

sync-config:
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/app-config.yaml | kubectl -n mlops apply -f -

ingest:
	$(MAKE) sync-config
	kubectl -n mlops delete job mlops-ingest --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/ingestion-job.yaml | kubectl -n mlops apply -f -

features:
	$(MAKE) sync-config
	kubectl -n mlops delete job mlops-features --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/feature-job.yaml | kubectl -n mlops apply -f -

train:
	$(MAKE) images
	$(MAKE) sync-config
	kubectl -n mlops delete job mlops-training --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/training-job.yaml | kubectl -n mlops apply -f -

serve:
	$(MAKE) images
	$(MAKE) sync-config
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/prometheus-config.yaml | kubectl -n mlops apply -f -
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/serving-deployment.yaml --show-only templates/serving-service.yaml | kubectl -n mlops apply -f -
	kubectl -n mlops rollout restart deployment/mlops-prometheus

drift:
	$(MAKE) sync-config
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/drift-cronjob.yaml | kubectl -n mlops apply -f -

drift-manual:
	$(MAKE) images
	$(MAKE) sync-config
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/drift-cronjob.yaml | kubectl -n mlops apply -f -
	kubectl -n mlops delete job drift-monitor-manual --ignore-not-found
	kubectl -n mlops create job --from=cronjob/mlops-drift-monitor drift-monitor-manual
	kubectl -n mlops wait --for=condition=complete job/drift-monitor-manual --timeout=300s
	kubectl -n mlops logs job/drift-monitor-manual --tail=120

pipeline-cron:
	$(MAKE) sync-config
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/pipeline-cronjob.yaml | kubectl -n mlops apply -f -

canary:
	$(MAKE) sync-config
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/serving-canary-deployment.yaml --show-only templates/serving-canary-service.yaml | kubectl -n mlops apply -f -

down:
	helm -n mlops uninstall $(HELM_RELEASE)
	minikube stop

reset-minikube:
	minikube delete
	minikube start --driver=docker --addons metrics-server --memory=$(MINIKUBE_MEMORY) --cpus=$(MINIKUBE_CPUS)

argo-apply:
	$(MAKE) argo-check
	kubectl apply -n $(ARGO_NAMESPACE) -f $(ARGO_APP)

argo-delete:
	$(MAKE) argo-check
	kubectl delete -n $(ARGO_NAMESPACE) -f $(ARGO_APP)

argo-sync:
	$(MAKE) argo-check
	argocd app sync $(ARGO_APP_NAME)

argo-get:
	$(MAKE) argo-check
	argocd app get $(ARGO_APP_NAME)

argo-check:
	@kubectl api-resources 2>/dev/null | grep -qi '^applications[[:space:]].*argoproj.io' || (echo 'ArgoCD CRDs are not installed in this cluster. Install ArgoCD before running argo-apply/argo-sync/argo-get.' && exit 1)
	@kubectl get namespace $(ARGO_NAMESPACE) >/dev/null 2>&1 || (echo 'Namespace $(ARGO_NAMESPACE) is missing. Install ArgoCD or create the namespace first.' && exit 1)
	@command -v argocd >/dev/null 2>&1 || echo 'Warning: argocd CLI not found in PATH. argo-sync and argo-get will fail unless the CLI is installed and pointed at a running ArgoCD server.'
