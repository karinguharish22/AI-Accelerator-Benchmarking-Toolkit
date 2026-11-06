# AccelBench workflow. Designed for Cloud Shell (gcloud, terraform, kubectl, envsubst preinstalled).
# Override any variable on the command line: make train-tpu PADDING=longest MAX_STEPS=100

SHELL := /bin/bash
PYTHON ?= python3

# Written by `make infra-up` from the Terraform outputs.
-include .env.generated

PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)
REGION     ?= us-central1
ZONE       ?= us-central1-a
CLUSTER    ?= accelbench
IMAGE_TAG  ?= latest

ifndef RUN_ID
RUN_ID := $(shell date +%Y%m%d-%H%M%S)
endif

# Experiment knobs
MODEL          ?= google/gemma-2b
SEQ_LEN        ?= 1024
MAX_STEPS      ?= 200
PADDING        ?= max_length
GPU_BATCH_SIZE ?= 4
TPU_BATCH_SIZE ?= 4
GPU_TYPE       ?= nvidia-l4
GPU_PROFILE    ?= gpu-l4x1
TPU_TRAIN_PROFILE ?= tpu-v5e-4
TPU_SERVE_PROFILE ?= tpu-v5e-1

# Serving / load test knobs
SERVE_MODEL    ?= $(MODEL)
SERVED_NAME    ?= accelbench-model
MAX_MODEL_LEN  ?= 2048
VLLM_GPU_IMAGE ?= vllm/vllm-openai:v0.6.3
VLLM_TPU_IMAGE ?= vllm/vllm-tpu:nightly
CONCURRENCY    ?= 1,4,16,32,64
INPUT_LEN      ?= 512
OUTPUT_LEN     ?= 256

SUBST_VARS := RUN_ID BUCKET REGISTRY IMAGE_TAG WORKLOAD_SA MODEL SEQ_LEN MAX_STEPS PADDING \
	GPU_BATCH_SIZE TPU_BATCH_SIZE GPU_TYPE GPU_PROFILE TPU_TRAIN_PROFILE TPU_SERVE_PROFILE \
	SERVE_MODEL SERVED_NAME MAX_MODEL_LEN VLLM_GPU_IMAGE VLLM_TPU_IMAGE CONCURRENCY INPUT_LEN \
	OUTPUT_LEN TARGET PROFILE ADAPTER_RUN

# $(call kapply,k8s/file.yaml): substitute only our ${VARS} and apply.
kapply = $(foreach v,$(SUBST_VARS),$(v)='$($(v))') envsubst '$(foreach v,$(SUBST_VARS),$${$(v)})' < $(1) | kubectl apply -f -

TF := terraform -chdir=terraform

.PHONY: help infra-up infra-down credentials build k8s-setup train-gpu train-tpu merge serve-gpu serve-tpu \
	bench-gpu bench-tpu status stop-serving clean-jobs fetch-results report demo dashboard tensorboard test lint

help: ## Show targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

require-%:
	@if [ -z "$($*)" ]; then echo "error: $* is not set (run make infra-up first, or pass $*=...)"; exit 1; fi

# --- Infrastructure -------------------------------------------------------------

infra-up: require-PROJECT_ID ## Create VPC, GKE cluster, GPU/TPU node pools, bucket, registry
	$(TF) init -input=false
	$(TF) apply -input=false -var project_id=$(PROJECT_ID) -var region=$(REGION) -var zone=$(ZONE)
	@echo "BUCKET := $$($(TF) output -raw bucket)" > .env.generated
	@echo "REGISTRY := $$($(TF) output -raw registry)" >> .env.generated
	@echo "WORKLOAD_SA := $$($(TF) output -raw workload_service_account)" >> .env.generated
	@cat .env.generated
	$(MAKE) credentials

infra-down: require-PROJECT_ID ## Destroy everything (run fetch-results first!)
	$(TF) destroy -input=false -var project_id=$(PROJECT_ID) -var region=$(REGION) -var zone=$(ZONE)
	rm -f .env.generated

credentials: ## Point kubectl at the cluster
	gcloud container clusters get-credentials $(CLUSTER) --zone $(ZONE) --project $(PROJECT_ID)

build: require-REGISTRY ## Build and push all images with Cloud Build
	gcloud builds submit --project $(PROJECT_ID) --config cloudbuild.yaml \
		--substitutions=_REGISTRY=$(REGISTRY),_TAG=$(IMAGE_TAG) .

k8s-setup: require-WORKLOAD_SA require-HF_TOKEN ## Namespace, service account, Hugging Face token secret
	$(call kapply,k8s/base.yaml)
	@# '@' keeps the token out of the terminal output.
	@kubectl -n accelbench create secret generic hf-token --from-literal=token="$$HF_TOKEN" \
		--dry-run=client -o yaml | kubectl apply -f -

# --- Experiments ----------------------------------------------------------------

train-gpu: require-BUCKET ## LoRA fine-tune on the GPU pool
	$(call kapply,k8s/train-gpu-job.yaml)
	@echo "follow: kubectl -n accelbench logs -f job/train-gpu-$(RUN_ID)"

train-tpu: require-BUCKET ## LoRA fine-tune on the TPU v5e-4 pool
	$(call kapply,k8s/train-tpu-job.yaml)
	@echo "follow: kubectl -n accelbench logs -f job/train-tpu-$(RUN_ID)"

merge: require-ADAPTER_RUN ## Merge an adapter (ADAPTER_RUN=<run-id>-<profile>) into the base model
	$(call kapply,k8s/merge-job.yaml)
	@echo "then serve it with: SERVE_MODEL=/results/models/$(ADAPTER_RUN)"

serve-gpu: require-BUCKET ## Start vLLM on the GPU pool
	$(call kapply,k8s/serve-gpu.yaml)
	kubectl -n accelbench rollout status deploy/vllm-gpu --timeout=40m

serve-tpu: require-BUCKET ## Start vLLM on the TPU v5e-1 pool
	$(call kapply,k8s/serve-tpu.yaml)
	kubectl -n accelbench rollout status deploy/vllm-tpu --timeout=50m

bench-gpu: TARGET := gpu
bench-gpu: PROFILE := $(GPU_PROFILE)
bench-gpu: require-BUCKET ## Load-test the GPU server
	$(call kapply,k8s/bench-job.yaml)
	@echo "follow: kubectl -n accelbench logs -f job/bench-gpu-$(RUN_ID)"

bench-tpu: TARGET := tpu
bench-tpu: PROFILE := $(TPU_SERVE_PROFILE)
bench-tpu: require-BUCKET ## Load-test the TPU server
	$(call kapply,k8s/bench-job.yaml)
	@echo "follow: kubectl -n accelbench logs -f job/bench-tpu-$(RUN_ID)"

status: ## Show pods, jobs and nodes
	kubectl -n accelbench get jobs,deploy,pods -o wide
	kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-spot

stop-serving: ## Delete model servers so accelerator nodes scale back to zero
	kubectl -n accelbench delete deploy,svc -l workload=serve --ignore-not-found

clean-jobs: ## Delete finished jobs
	kubectl -n accelbench delete jobs -l app=accelbench --ignore-not-found

# --- Analysis (runs locally) ----------------------------------------------------

fetch-results: require-BUCKET ## Download results from the bucket into ./results
	mkdir -p results
	gcloud storage rsync -r --exclude='^models/' gs://$(BUCKET) results

report: ## Build results/report (tables, charts, recommendation)
	$(PYTHON) -m analysis.report --results results

demo: ## Generate SYNTHETIC sample data and a report, to try the pipeline for free
	$(PYTHON) -m analysis.make_demo_data --out results-demo
	$(PYTHON) -m analysis.report --results results-demo

dashboard: ## Open the cost/performance advisor
	streamlit run dashboard/app.py -- --results results

tensorboard: ## Loss curves and profiler traces
	tensorboard --logdir results/training

test: ## Unit tests
	$(PYTHON) -m pytest -q

lint:
	ruff check .
	terraform fmt -check -recursive terraform
