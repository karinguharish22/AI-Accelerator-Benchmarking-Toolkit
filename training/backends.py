"""Device backends: every line of accelerator-specific code in the trainer lives here.

`train.py` is written once against the `Backend` interface. Porting a workload
from CUDA to TPU (PyTorch/XLA) means implementing these few methods, which is
the core of the GPU -> TPU migration path described in docs/migration-playbook.md.
"""

from __future__ import annotations

import contextlib
import os


class Backend:
    name: str
    device = None
    world_size: int = 1
    rank: int = 0

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def wrap_model(self, model):
        return model

    def wrap_loader(self, loader):
        return loader

    def to_device(self, batch: dict) -> dict:
        return batch

    def optimizer_step(self, optimizer) -> None:
        optimizer.step()

    def sync(self) -> None:
        """Block until queued device work finishes, so wall-clock timings are honest."""

    def barrier(self, tag: str) -> None:
        pass

    def step_context(self, step: int):
        return contextlib.nullcontext()

    def start_profile(self, logdir: str, num_steps: int, est_step_time_s: float) -> None:
        pass

    def profile_step(self) -> None:
        pass

    def memory_gb(self) -> float | None:
        return None

    def compile_stats(self) -> dict:
        return {}

    def to_cpu_state_dict(self, state_dict: dict) -> dict:
        return {k: v.detach().cpu() for k, v in state_dict.items()}


class CpuBackend(Backend):
    """Smoke tests only: checks the training loop end to end with a tiny model, no accelerator."""

    name = "cpu"

    def __init__(self):
        import torch

        self.device = torch.device("cpu")


class CudaBackend(Backend):
    """NVIDIA GPUs. Multi-GPU runs use DDP when launched with torchrun."""

    name = "cuda"

    def __init__(self):
        import torch
        import torch.distributed as dist

        self._torch = torch
        self._dist = dist
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world_size > 1:
            dist.init_process_group("nccl")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._profiler = None
        self._profile_steps_left = 0

    def wrap_model(self, model):
        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel

            return DistributedDataParallel(model, device_ids=[self.local_rank])
        return model

    def to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def sync(self):
        self._torch.cuda.synchronize()

    def barrier(self, tag):
        if self.world_size > 1:
            self._dist.barrier()

    def start_profile(self, logdir, num_steps, est_step_time_s):
        from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

        self._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=num_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(logdir),
            record_shapes=True,
            profile_memory=True,
        )
        self._profiler.__enter__()
        self._profile_steps_left = num_steps + 1

    def profile_step(self):
        if self._profiler is None:
            return
        self._profiler.step()
        self._profile_steps_left -= 1
        if self._profile_steps_left <= 0:
            self._profiler.__exit__(None, None, None)
            self._profiler = None

    def memory_gb(self):
        return self._torch.cuda.max_memory_allocated(self.device) / 1e9


class XlaBackend(Backend):
    """Cloud TPUs via PyTorch/XLA. One process per TPU chip, started by xmp.spawn."""

    name = "xla"
    PROFILER_PORT = 9012

    def __init__(self):
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.debug.profiler as xp
        import torch_xla.runtime as xr

        self._xm, self._met, self._xp = xm, met, xp
        self.device = xm.xla_device()
        self.world_size = xr.world_size()
        self.rank = xr.global_ordinal()
        self._profiler_server = xp.start_server(self.PROFILER_PORT) if self.is_master else None

    def wrap_loader(self, loader):
        # Overlaps host->device transfer with compute and calls mark_step() per batch.
        import torch_xla.distributed.parallel_loader as pl

        return pl.MpDeviceLoader(loader, self.device)

    def optimizer_step(self, optimizer):
        # All-reduces gradients across chips, then steps.
        self._xm.optimizer_step(optimizer)

    def sync(self):
        self._xm.mark_step()
        self._xm.wait_device_ops()

    def barrier(self, tag):
        self._xm.rendezvous(tag)

    def step_context(self, step):
        # Names the step in the trace viewer and cuts the lazy graph at step end.
        return self._xp.StepTrace("train_step", step_num=step)

    def start_profile(self, logdir, num_steps, est_step_time_s):
        if not self.is_master:
            return
        duration_ms = int(max(3.0, est_step_time_s * num_steps * 1.5) * 1000)
        self._xp.trace_detached(f"localhost:{self.PROFILER_PORT}", logdir, duration_ms=duration_ms)

    def memory_gb(self):
        try:
            info = self._xm.get_memory_info(self.device)
        except Exception:
            return None
        if "bytes_used" in info:
            return info["bytes_used"] / 1e9
        if "kb_total" in info:
            return (info["kb_total"] - info["kb_free"]) / 1e6
        return None

    def compile_stats(self):
        stats = {}
        compile_data = self._met.metric_data("CompileTime")
        if compile_data:
            stats["xla_compilations"] = compile_data[0]
            stats["xla_compile_time_s"] = compile_data[1] / 1e9
        execute_data = self._met.metric_data("ExecuteTime")
        if execute_data:
            stats["xla_executions"] = execute_data[0]
        return stats

    def metrics_report(self) -> str:
        return self._met.metrics_report()

    def to_cpu_state_dict(self, state_dict):
        self.sync()
        return {k: v.detach().to("cpu") for k, v in state_dict.items()}


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return "xla"
    return "cuda"


def make_backend(device: str) -> Backend:
    if device == "cpu":
        return CpuBackend()
    if device == "cuda":
        return CudaBackend()
    if device == "xla":
        return XlaBackend()
    raise ValueError(f"unsupported device {device!r}")
