# Profiling guide

Every training run captures a trace of `--profile-steps` steps starting at
`--profile-start-step` (default: 5 steps from step 50), written to
`results/training/<run>/profile/`.

## Viewing traces

```bash
make fetch-results
pip install tensorboard torch-tb-profiler tensorboard-plugin-profile
tensorboard --logdir results/training
```

- **GPU runs:** the *PyTorch Profiler* tab (from `torch-tb-profiler`) shows the overview,
  operator, kernel and memory views and the trace timeline.
- **TPU runs:** the *Profile* tab (from `tensorboard-plugin-profile`) shows the overview page,
  trace viewer, op profile and memory viewer.
- **TPU, any run:** `xla_metrics_report.txt` lists `CompileTime` counts and
  `aten::` ops that fell back to CPU.

## What to look for

| Symptom in the trace | Diagnosis | Fix to try (flag) |
|---|---|---|
| TPU: `CompileTime` count keeps growing, long gaps between steps | Dynamic shapes cause recompilation | `--padding max_length` or `--padding bucket` |
| GPU or TPU: device idle while the host prepares batches | Input-bound | `--num-workers 4`, pre-tokenize |
| Low MFU, short kernels, gaps between them | Too little work per step | Raise `--per-device-batch-size` |
| Out of memory at a larger batch size | Activation memory | `--gradient-checkpointing` (trades about 30% compute for memory) |
| TPU: frequent small `TransferFromDevice` | Host syncs inside the loop | Read tensors only at `--log-every` |
| GPU: large `aten::copy_`/`cudaMemcpy` | Unpinned host memory | `pin_memory=True` (default here) |

## Experiments that give you a story

Run each pair and record the numbers in [experiment-log.md](experiment-log.md):

1. **XLA recompilation.**
   `make train-tpu PADDING=longest` vs `make train-tpu PADDING=bucket` vs `PADDING=max_length`.
   Compare `xla_compilations`, `first_step_s` and steady-state `tokens_per_s`.
2. **Batch-size scaling.** `TPU_BATCH_SIZE=2,4,8,16` and `GPU_BATCH_SIZE=2,4,8`. Plot
   tokens/s per chip and MFU, and find where each accelerator saturates or runs out of memory.
3. **Memory vs compute.** The largest batch that fits with and without gradient checkpointing.
4. **Serving concurrency.** Find the knee in the latency-throughput frontier chart for each
   accelerator, which is where p99 TPOT starts climbing sharply.

Only claim an improvement you measured, and keep the before and after `metrics.json`.
