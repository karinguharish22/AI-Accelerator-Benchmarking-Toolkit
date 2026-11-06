# Experiment log

Fill this in as you run experiments. One row per run, with values copied from `metrics.json`
or `results/report/*.csv`.

## Environment

| Item | Value |
|---|---|
| Date | |
| Region / zone | |
| GKE version | |
| Training images (digest) | |
| vLLM GPU image | |
| vLLM TPU image | |
| Model | |
| Pricing source and date | |

## Training

| Run ID | Profile | Padding | Batch/chip | Tokens/s | Tokens/s/chip | MFU | XLA compiles | First step (s) | $/1M tokens | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

## Serving

| Run ID | Profile | Concurrency | Out tok/s | TTFT p99 (ms) | TPOT p99 (ms) | $/1M out tokens | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Findings

1. **Hypothesis:**
   **Change:**
   **Result (before → after):**
   **Evidence:** trace screenshot / metrics file
