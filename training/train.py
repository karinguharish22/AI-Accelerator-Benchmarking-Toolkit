"""LoRA fine-tuning benchmark that runs unchanged on NVIDIA GPUs and Cloud TPUs.

GPU:  python -m training.train --device cuda ...      (torchrun for multi-GPU)
TPU:  PJRT_DEVICE=TPU python -m training.train --device xla ...  (one process per chip)

Writes metrics.json (throughput, step time, memory, XLA compile stats, loss curve),
TensorBoard scalars, an optional profiler trace and optionally the LoRA adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone

from training.data import PADDING_MODES, build_dataset, make_collator
from training.flops import train_flops_per_token


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["auto", "cuda", "xla", "cpu"], default="auto",
                   help="cpu is for smoke tests with a tiny model.")
    p.add_argument("--accelerator-profile", default=os.environ.get("ACCEL_PROFILE", "unknown"),
                   help="Key into configs/pricing.yaml, e.g. gpu-l4x1 or tpu-v5e-4.")
    p.add_argument("--run-id", default=os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--model", default="google/gemma-2b")
    p.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--padding", choices=PADDING_MODES, default="max_length",
                   help="max_length = static shapes (XLA-friendly); longest = dynamic shapes.")
    p.add_argument("--bucket-size", type=int, default=128)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="LR warmup steps; also excluded from steady-state throughput.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--attn-implementation", default=None, help="Defaults to sdpa on GPU, eager on TPU.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--profile-start-step", type=int, default=50, help="0 disables profiling.")
    p.add_argument("--profile-steps", type=int, default=5)
    p.add_argument("--output-dir", default="results/training")
    p.add_argument("--save-adapter", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def run(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    from training.backends import make_backend

    backend = make_backend(args.device)
    torch.manual_seed(args.seed)
    out_dir = os.path.join(args.output_dir, f"{args.run_id}-{args.accelerator_profile}")
    if backend.is_master:
        os.makedirs(out_dir, exist_ok=True)
        print(f"[accelbench] backend={backend.name} world_size={backend.world_size} out={out_dir}", flush=True)

    # Rank 0 downloads model + dataset first; the other ranks then load from the shared cache.
    if not backend.is_master:
        backend.barrier("download")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn = args.attn_implementation or ("eager" if backend.name == "xla" else "sdpa")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation=attn)
    dataset = build_dataset(tokenizer, args.dataset, args.seq_len, args.max_examples)
    if backend.is_master:
        backend.barrier("download")

    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peft_model = model
    model = backend.wrap_model(model.to(backend.device))

    sampler = DistributedSampler(
        dataset, num_replicas=backend.world_size, rank=backend.rank, shuffle=True, seed=args.seed, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        collate_fn=make_collator(tokenizer.pad_token_id, args.padding, args.seq_len, args.bucket_size),
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=backend.name == "cuda",
    )
    device_loader = backend.wrap_loader(loader)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    writer = None
    if backend.is_master:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))

    model.train()
    step, micro, epoch, tokens = 0, 0, 0, 0
    loss_curve = []
    last_loss = None
    t_start = time.perf_counter()
    first_step_s = None
    t_steady, tokens_steady = None, 0
    t_window, tokens_window = t_start, 0

    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in device_loader:
            batch = backend.to_device(batch)
            with backend.step_context(micro):
                loss = model(**batch).loss / args.grad_accum
                loss.backward()
                micro += 1
                # Shape is metadata, so reading it never forces an XLA sync.
                tokens += batch["input_ids"].numel() * backend.world_size
                if micro % args.grad_accum == 0:
                    backend.optimizer_step(optimizer)
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    last_loss = loss

            if micro % args.grad_accum != 0:
                continue

            if step == 1:
                backend.sync()
                # Includes graph compilation on XLA and kernel autotuning on CUDA.
                first_step_s = time.perf_counter() - t_start
            if step == args.warmup_steps:
                backend.sync()
                t_steady, tokens_steady = time.perf_counter(), tokens

            if args.profile_start_step and step == args.profile_start_step:
                est = (time.perf_counter() - t_steady) / max(1, step - args.warmup_steps) if t_steady else 1.0
                backend.start_profile(os.path.join(out_dir, "profile"), args.profile_steps, est)
            backend.profile_step()

            if step % args.log_every == 0:
                backend.sync()
                now = time.perf_counter()
                tps = (tokens - tokens_window) / (now - t_window)
                t_window, tokens_window = now, tokens
                loss_value = last_loss.item() * args.grad_accum
                loss_curve.append({"step": step, "loss": loss_value, "tokens_per_s": tps})
                if backend.is_master:
                    print(f"[accelbench] step {step}/{args.max_steps} loss={loss_value:.4f} tok/s={tps:,.0f}",
                          flush=True)
                    writer.add_scalar("train/loss", loss_value, step)
                    writer.add_scalar("train/tokens_per_s", tps, step)

            if step >= args.max_steps:
                break
        epoch += 1

    backend.sync()
    t_end = time.perf_counter()
    if t_steady is None:
        raise SystemExit("--max-steps must be larger than --warmup-steps to measure steady-state throughput")
    steady_steps = args.max_steps - args.warmup_steps
    steady_tps = (tokens - tokens_steady) / (t_end - t_steady)

    if backend.is_master:
        writer.close()
        metrics = {
            "kind": "training",
            "run_id": args.run_id,
            "accelerator_profile": args.accelerator_profile,
            "backend": backend.name,
            "num_devices": backend.world_size,
            "model": args.model,
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "padding": args.padding,
            "per_device_batch_size": args.per_device_batch_size,
            "grad_accum": args.grad_accum,
            "global_batch_size": args.per_device_batch_size * args.grad_accum * backend.world_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": attn,
            "lora_r": args.lora_r,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops_per_token": train_flops_per_token(total_params, trainable_params),
            "tokens_per_s": steady_tps,
            "tokens_per_s_per_device": steady_tps / backend.world_size,
            "step_time_s": (t_end - t_steady) / steady_steps,
            "first_step_s": first_step_s,
            "total_wall_s": t_end - t_start,
            "tokens_trained": tokens,
            "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
            "memory_gb": backend.memory_gb(),
            **backend.compile_stats(),
            "loss_curve": loss_curve,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if hasattr(backend, "metrics_report"):
            with open(os.path.join(out_dir, "xla_metrics_report.txt"), "w") as f:
                f.write(backend.metrics_report())
        print(f"[accelbench] done: {steady_tps:,.0f} tok/s steady-state, metrics in {out_dir}", flush=True)

        if args.save_adapter:
            state = backend.to_cpu_state_dict(get_peft_model_state_dict(peft_model))
            peft_model.save_pretrained(os.path.join(out_dir, "adapter"), state_dict=state)
            tokenizer.save_pretrained(os.path.join(out_dir, "adapter"))

    backend.barrier("done")


def _xla_worker(index, args):
    run(args)


def main(argv=None):
    from training.backends import resolve_device

    args = parse_args(argv)
    args.device = resolve_device(args.device)
    if args.device == "xla":
        import torch_xla.distributed.xla_multiprocessing as xmp

        xmp.spawn(_xla_worker, args=(args,))
    else:
        run(args)


if __name__ == "__main__":
    main()
