# Setup and run guide

Everything below runs from **Cloud Shell** (console.cloud.google.com → the `>_` icon), which
already has `gcloud`, `terraform`, `kubectl`, `make`, `envsubst` and Python. Nothing needs a
local GPU or local Docker.

## 0. One-time prerequisites

1. **Billing.** Free Trial accounts cannot use GPUs or TPUs. Upgrade to a paid account in
   *Billing → Account management* (the remaining trial credit is kept and still applies).
2. **Quota.** In *IAM & Admin → Quotas*, request for your region (default `us-central1`):
   - `Preemptible NVIDIA L4 GPUs` ≥ 1 (or `NVIDIA L4 GPUs` if `use_spot = false`)
   - `TPU v5 Lite PodSlice chips` (preemptible) ≥ 5 (4 for training + 1 for serving)

   Approval can take from minutes to a couple of days, so request quota first.
3. **Model access.** Gemma is gated. Accept the license at
   <https://huggingface.co/google/gemma-2b>, then create a read token at
   <https://huggingface.co/settings/tokens>.
4. **Check the zone.** Both `nvidia-l4` and TPU v5e must exist in your zone:
   ```bash
   gcloud compute accelerator-types list --filter="zone:us-central1-a AND name:nvidia-l4"
   gcloud compute tpus accelerator-types list --zone=us-central1-a | grep v5litepod
   ```
5. **Set a budget alert** in *Billing → Budgets & alerts* (for example $50), so a forgotten
   node cannot surprise you.

## 1. Infrastructure (~10 min)

```bash
git clone https://github.com/karinguharish22/AI-Accelerator-Benchmarking-Toolkit.git
cd AI-Accelerator-Benchmarking-Toolkit
gcloud config set project <PROJECT_ID>
make infra-up            # Terraform: VPC, GKE, GPU + 2 TPU pools (all scale to 0), bucket, registry
make build               # Cloud Build: train-gpu, train-tpu, tools images (~15-25 min the first time)
export HF_TOKEN=hf_xxx   # do not commit this
make k8s-setup           # namespace, Workload Identity service account, HF token secret
```

## 2. Fine-tuning benchmarks

Accelerator nodes start from zero, so the first pod waits a few minutes for a node.

```bash
make train-gpu                      # 1x L4
make train-tpu                      # TPU v5e-4
make status                         # watch pods / nodes
kubectl -n accelbench logs -f job/train-tpu-<RUN_ID>
```

Experiments worth running (each writes a separate result):

```bash
make train-tpu PADDING=longest      # dynamic shapes -> XLA recompilation (see profiling guide)
make train-tpu PADDING=bucket       # the fix: a handful of shapes
make train-gpu GPU_BATCH_SIZE=8     # batch-size scaling
```

## 3. Serving benchmarks

```bash
make serve-gpu && make bench-gpu
make serve-tpu && make bench-tpu
make stop-serving                   # IMPORTANT: lets the accelerator nodes scale back to zero
```

To serve your fine-tuned model instead of the base model:

```bash
make merge ADAPTER_RUN=<run-id>-gpu-l4x1
make serve-gpu SERVE_MODEL=/results/models/<run-id>-gpu-l4x1
```

## 4. Analyze

```bash
make fetch-results                  # bucket -> ./results
pip install -r requirements-dev.txt
make report                         # results/report/report.md + charts
make dashboard                      # Cloud Shell: use Web Preview on port 8501
make tensorboard                    # loss curves + profiler traces
```

Before trusting any cost number, put current prices for your zone into `configs/pricing.yaml`.

## 5. Tear down

```bash
make stop-serving clean-jobs
make fetch-results                  # the bucket is deleted with the infra
make infra-down
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Pod `Pending`, event `FailedScaleUp ... quota` | GPU/TPU quota not granted in this region |
| Pod `Pending`, `GCE out of resources` | Spot capacity is short right now; retry later, try another zone, or set `use_spot = false` |
| `401` / `gated repo` in logs | HF token missing, or Gemma license not accepted for that HF account |
| `Permission denied` writing `/results` | `make k8s-setup` not run, or the Workload Identity binding is still propagating (wait 1-2 min) |
| Spot pod killed mid-run | Expected with Spot. The Job retries (`backoffLimit: 2`) |
| vLLM TPU pod restarts during startup | Precompilation took longer than the startup probe allows: raise `failureThreshold` in `k8s/serve-tpu.yaml` |
