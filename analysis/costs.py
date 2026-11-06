"""Cost-to-performance model shared by the report generator and the dashboard."""

from __future__ import annotations

import glob
import json
import os

import yaml

from training.flops import mfu

PRICING_MODES = ("spot", "on_demand")


def load_pricing(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["profiles"]


def hourly_price(pricing: dict, profile: str, mode: str) -> float | None:
    entry = pricing.get(profile)
    if entry is None:
        return None
    return entry["hourly_usd"][mode]


def cost_per_million_tokens(hourly_usd: float | None, tokens_per_s: float) -> float | None:
    if hourly_usd is None or tokens_per_s <= 0:
        return None
    return hourly_usd / (tokens_per_s * 3600) * 1e6


def _load_json_files(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = path
        out.append(data)
    return out


def load_training_runs(results_dir: str) -> list[dict]:
    return _load_json_files(os.path.join(results_dir, "training", "**", "metrics.json"))


def load_inference_runs(results_dir: str) -> list[dict]:
    return _load_json_files(os.path.join(results_dir, "inference", "**", "summary.json"))


def is_synthetic(*runs: list[dict]) -> bool:
    return any(r.get("synthetic") for group in runs for r in group)


def training_rows(runs: list[dict], pricing: dict, mode: str) -> list[dict]:
    rows = []
    for r in runs:
        profile = r["accelerator_profile"]
        price = hourly_price(pricing, profile, mode)
        peak = pricing.get(profile, {}).get("peak_bf16_tflops_per_chip", 0)
        rows.append({
            "run_id": r["run_id"],
            "accelerator_profile": profile,
            "model": r["model"],
            "num_devices": r["num_devices"],
            "seq_len": r["seq_len"],
            "padding": r["padding"],
            "global_batch_size": r["global_batch_size"],
            "tokens_per_s": r["tokens_per_s"],
            "tokens_per_s_per_device": r["tokens_per_s_per_device"],
            "step_time_s": r["step_time_s"],
            "first_step_s": r.get("first_step_s"),
            "mfu": mfu(r["tokens_per_s_per_device"], r["flops_per_token"], peak),
            "xla_compilations": r.get("xla_compilations"),
            "memory_gb": r.get("memory_gb"),
            "final_loss": r.get("final_loss"),
            "hourly_usd": price,
            "usd_per_million_tokens": cost_per_million_tokens(price, r["tokens_per_s"]),
            "usd_per_run": price * r["total_wall_s"] / 3600 if price is not None else None,
        })
    return rows


def inference_rows(runs: list[dict], pricing: dict, mode: str) -> list[dict]:
    rows = []
    for r in runs:
        profile = r["accelerator_profile"]
        price = hourly_price(pricing, profile, mode)
        for level in r["levels"]:
            rows.append({
                "run_id": r["run_id"],
                "accelerator_profile": profile,
                "model": r["model"],
                "input_len": r["input_len"],
                "output_len": r["output_len"],
                **{k: v for k, v in level.items() if k != "wall_s"},
                "hourly_usd": price,
                "usd_per_million_output_tokens": cost_per_million_tokens(price, level["output_tokens_per_s"]),
            })
    return rows


def recommend(rows: list[dict], max_ttft_p99_ms: float | None = None, max_tpot_p99_ms: float | None = None,
              min_output_tokens_per_s: float | None = None) -> list[dict]:
    """Operating points that meet every latency/throughput constraint, cheapest first."""

    def ok(row):
        if row.get("failed"):
            return False
        if row.get("usd_per_million_output_tokens") is None:
            return False
        if max_ttft_p99_ms is not None and (row["ttft_ms_p99"] is None or row["ttft_ms_p99"] > max_ttft_p99_ms):
            return False
        if max_tpot_p99_ms is not None and (row["tpot_ms_p99"] is None or row["tpot_ms_p99"] > max_tpot_p99_ms):
            return False
        if min_output_tokens_per_s is not None and row["output_tokens_per_s"] < min_output_tokens_per_s:
            return False
        return True

    return sorted(filter(ok, rows), key=lambda r: r["usd_per_million_output_tokens"])


def best_per_profile(rows: list[dict]) -> list[dict]:
    """Cheapest feasible operating point for each accelerator profile."""
    best = {}
    for row in rows:
        current = best.get(row["accelerator_profile"])
        if current is None or row["usd_per_million_output_tokens"] < current["usd_per_million_output_tokens"]:
            best[row["accelerator_profile"]] = row
    return sorted(best.values(), key=lambda r: r["usd_per_million_output_tokens"])
