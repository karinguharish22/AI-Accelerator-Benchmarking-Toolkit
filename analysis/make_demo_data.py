"""Write SYNTHETIC results so the report and dashboard can be tried without any cloud spend.

Every file is tagged "synthetic": true, and the report and dashboard print a warning
banner for it. The numbers are made up: never publish them or put them on a resume.

python -m analysis.make_demo_data --out results-demo
"""

import argparse
import json
import os
import random

PROFILES = {
    # profile: (chips, relative training speed per chip, relative serving speed)
    "gpu-l4x1": (1, 1.0, 1.0),
    "tpu-v5e-4": (4, 1.4, None),
    "tpu-v5e-1": (1, None, 1.3),
}


def training_run(profile, chips, speed, padding, rng):
    per_chip = 2000 * speed * (0.55 if padding == "longest" and profile.startswith("tpu") else 1.0)
    tps = per_chip * chips * rng.uniform(0.95, 1.05)
    steps = 200
    return {
        "kind": "training", "synthetic": True, "run_id": f"demo-{padding}", "accelerator_profile": profile,
        "backend": "xla" if profile.startswith("tpu") else "cuda", "num_devices": chips,
        "model": "google/gemma-2b", "dataset": "databricks/databricks-dolly-15k", "seq_len": 1024,
        "padding": padding, "per_device_batch_size": 4, "grad_accum": 1, "global_batch_size": 4 * chips,
        "total_params": 2_506_172_416, "trainable_params": 3_686_400, "flops_per_token": 1.0e10,
        "tokens_per_s": tps, "tokens_per_s_per_device": tps / chips, "step_time_s": 4 * chips * 1024 / tps,
        "first_step_s": rng.uniform(60, 120) if profile.startswith("tpu") else rng.uniform(5, 15),
        "total_wall_s": steps * 4 * chips * 1024 / tps + 90,
        "xla_compilations": (rng.randint(6, 9) if padding == "longest" else 2) if profile.startswith("tpu") else None,
        "final_loss": rng.uniform(1.2, 1.5), "loss_curve": [],
    }


def inference_run(profile, speed, rng):
    levels = []
    for c in (1, 4, 16, 32, 64):
        tps = 30 * speed * c / (1 + c / 24) * rng.uniform(0.95, 1.05)
        tpot = 1000 * c / tps
        levels.append({
            "concurrency": c, "requests": max(32, 4 * c), "succeeded": max(32, 4 * c), "failed": 0, "wall_s": 60.0,
            "mean_prompt_tokens": 512, "output_tokens_per_s": tps, "requests_per_s": tps / 256,
            "ttft_ms_p50": 40 * (1 + c / 8) / speed, "ttft_ms_p99": 90 * (1 + c / 6) / speed,
            "tpot_ms_p50": tpot, "tpot_ms_p99": tpot * 1.3,
            "e2e_s_p50": tpot * 256 / 1000, "e2e_s_p99": tpot * 1.3 * 256 / 1000,
        })
    return {"kind": "inference", "synthetic": True, "run_id": "demo", "accelerator_profile": profile,
            "server": "vllm", "model": "google/gemma-2b", "input_len": 512, "output_len": 256, "levels": levels}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results-demo")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)

    for profile, (chips, train_speed, serve_speed) in PROFILES.items():
        if train_speed:
            for padding in ("max_length", "longest"):
                d = os.path.join(args.out, "training", f"demo-{padding}-{profile}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "metrics.json"), "w") as f:
                    json.dump(training_run(profile, chips, train_speed, padding, rng), f, indent=2)
        if serve_speed:
            d = os.path.join(args.out, "inference", f"demo-{profile}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "summary.json"), "w") as f:
                json.dump(inference_run(profile, serve_speed, rng), f, indent=2)
    print(f"SYNTHETIC demo results written to {args.out}")


if __name__ == "__main__":
    main()
