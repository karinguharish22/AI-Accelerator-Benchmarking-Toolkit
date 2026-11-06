# GPU → TPU migration playbook (PyTorch LLM workloads)

A phased path for moving a PyTorch fine-tuning or serving workload from NVIDIA GPUs to
Cloud TPU on GKE, based on the port done in this repository.

## Phase 1: Assess (1-2 days)

| Question | Good fit for TPU | Proceed with care |
|---|---|---|
| Model architecture | Standard transformers (Llama, Gemma, Mistral, T5, ViT) | Custom CUDA kernels, unusual ops without XLA lowering |
| Input shapes | Fixed or bucketable sequence lengths | Highly variable shapes that cannot be padded |
| Control flow | Static graph per step | Data-dependent Python branching inside `forward` |
| Precision | bf16 | Workloads that depend on fp16 loss scaling or fp8 GPU kernels |
| Scale | Data parallel to multi-host pods | — |
| Serving | vLLM- or JetStream-supported models | Exotic decoding strategies |

Run `PT_XLA_DEBUG=1` on a short run to list ops that fall back to CPU.

## Phase 2: Port the code

All the device-specific changes in this repo are in [`training/backends.py`](../training/backends.py):

| Concern | CUDA | PyTorch/XLA |
|---|---|---|
| Device | `torch.device("cuda", local_rank)` | `xm.xla_device()` |
| Process launch | `torchrun --nproc-per-node N` | `xmp.spawn(fn)` (one process per chip) |
| Input pipeline | `DataLoader(pin_memory=True)` + `.to(device)` | `pl.MpDeviceLoader(loader, device)` (prefetch + `mark_step`) |
| Gradient sync + step | DDP wrapper + `optimizer.step()` | `xm.optimizer_step(optimizer)` |
| Timing / sync | `torch.cuda.synchronize()` | `xm.wait_device_ops()` |
| Profiling | `torch.profiler` | `xp.start_server` + `xp.trace_detached`, `xp.StepTrace` |
| Barrier | `dist.barrier()` | `xm.rendezvous(tag)` |
| Save | `torch.save` | move tensors to CPU first (`.to("cpu")`) |

Rules that matter more on TPU than on GPU:

1. **Static shapes.** Every new input shape triggers an XLA compile. Pad to a fixed length or
   to buckets (`--padding max_length|bucket`). Measure with `xla_compilations` in
   `metrics.json`.
2. **No host syncs in the hot loop.** `loss.item()`, `print(tensor)` and `if tensor > 0:` cut
   the graph and stall the device. Read values only at log intervals.
3. **bf16 everywhere.** TPU MXUs are designed for bf16, and bf16 needs no loss scaling.
4. **Use eager or a TPU-supported attention implementation.** Some GPU-oriented attention
   implementations do not lower cleanly to XLA.
5. **Larger per-chip batches.** TPUs usually need more work per step to keep the MXUs busy.

## Phase 3: Validate correctness

- Same seed, data and hyperparameters on both backends. The loss curves should track each
  other within normal bf16 noise (compare `loss_curve` in the two `metrics.json` files, or in
  TensorBoard).
- Evaluate both adapters on the same held-out prompts.

## Phase 4: Optimize and benchmark

Follow [profiling-guide.md](profiling-guide.md). Typical order of wins on TPU:
recompilation → input pipeline → batch size → host syncs → sharding strategy.

## Phase 5: Productionize on GKE

- **Node pools:** single-host v5e (`ct5lp-hightpu-{1,4,8}t`) for small models;
  multi-host slices with `placement_policy` / topology for larger ones (JobSet for
  multi-host training).
- **Obtaining capacity:** Spot for experiments, on-demand or reservations for SLAs, and
  Dynamic Workload Scheduler (flex-start) for batch training that can wait.
- **Serving:** vLLM TPU or JetStream behind a GKE Service / Gateway. Precompilation makes
  startup slower than on GPU, so set the startup probes and scaling policy for it.
- **Scaling out:** for models beyond about 8B parameters, move from data parallel to
  FSDPv2/SPMD in PyTorch/XLA, or to JAX with MaxText.

## Phase 6: Decide with data

Use `make report` / the dashboard to answer three questions:

1. Cost per 1M training tokens on each platform.
2. Cheapest serving configuration that meets the latency SLA.
3. Engineering cost of the port (this playbook plus the measured validation effort).

## Rollout checklist

- [ ] Quotas approved in the target region(s)
- [ ] Container images pinned by digest
- [ ] Loss parity validated
- [ ] Serving SLA validated at expected peak concurrency
- [ ] Checkpointing to GCS tested against Spot preemption
- [ ] Monitoring: TPU duty cycle / HBM usage in Cloud Monitoring dashboards
- [ ] Rollback path to the GPU deployment kept warm during cutover
