HELM_RELEASE=mlops-shared-volume
CHART_DIR=helm/shared-volume
SHARED_PATH=/tmp/mlops-shared
HELM_SET=--set sharedVolume.useHostPath=true --set sharedVolume.hostPath=$(SHARED_PATH)

.PHONY: up ingest features train serve down images build-images

up:
	minikube start --driver=docker --addons metrics-server
	minikube ssh -- "sudo mkdir -p $(SHARED_PATH) && sudo chown docker:docker $(SHARED_PATH)"
	kubectl -n mlops delete job mlops-ingest mlops-features mlops-training --ignore-not-found
	kubectl -n mlops delete deployment mlops-serving mlops-redis --ignore-not-found
	helm upgrade --install $(HELM_RELEASE) $(CHART_DIR) --create-namespace --namespace mlops $(HELM_SET)

images: build-images

build-images:
	docker build -t mlops-training:latest apps/training
	minikube image load mlops-training:latest
	docker build -t mlops-serving:latest -f apps/serving/Dockerfile .
	minikube image load mlops-serving:latest

ingest:
	kubectl -n mlops delete job mlops-ingest --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/ingestion-job.yaml | kubectl -n mlops apply -f -

features:
	kubectl -n mlops delete job mlops-features --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/feature-job.yaml | kubectl -n mlops apply -f -

train:
	kubectl -n mlops delete job mlops-training --ignore-not-found
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/training-job.yaml | kubectl -n mlops apply -f -

serve:
	helm template mlops-shared-volume $(CHART_DIR) --namespace mlops $(HELM_SET) --show-only templates/serving-deployment.yaml | kubectl -n mlops apply -f -

down:
	helm -n mlops uninstall $(HELM_RELEASE)
	minikube stop
