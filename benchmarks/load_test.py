"""Concurrency-sweep load test for any OpenAI-compatible completions server (vLLM on GPU or TPU).

For each concurrency level it fires a fixed number of streaming requests with a
fixed prompt length and a fixed output length (ignore_eos), then records:
  - TTFT   time to first token
  - TPOT   time per output token after the first
  - E2E    end-to-end request latency
  - output tokens/s and requests/s across the whole level

python -m benchmarks.load_test --base-url http://vllm-gpu:8000 --model google/gemma-2b \
    --accelerator-profile gpu-l4x1 --concurrency 1,4,16,32,64 --output results/inference/run1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

WORDS = (
    "cloud accelerator tensor model training inference latency throughput cluster kernel memory "
    "bandwidth compiler graph token batch layer attention vector matrix gradient optimizer schedule "
    "network storage pipeline benchmark profile cost region zone node pool container service"
).split()


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    error: str | None = None
    concurrency: int = field(default=0)

    @property
    def tpot_s(self) -> float | None:
        if not self.ok or self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)


def make_prompt(approx_tokens: int, rng: random.Random) -> str:
    # Roughly one token per word; the server reports the exact prompt length in `usage`.
    return " ".join(rng.choice(WORDS) for _ in range(approx_tokens))


def parse_sse_line(line: str) -> dict | None:
    """Parse one Server-Sent-Events line. Returns None for keep-alives and the [DONE] marker."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None
    return json.loads(data)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(results: list[RequestResult], concurrency: int, wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e = [r.e2e_s for r in ok]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    out_tokens = sum(r.output_tokens for r in ok)
    prompt_tokens = [r.prompt_tokens for r in ok if r.prompt_tokens]
    ms = 1000.0
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "wall_s": wall_s,
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else None,
        "output_tokens_per_s": out_tokens / wall_s if wall_s > 0 else 0.0,
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        "ttft_ms_p50": _scale(percentile(ttft, 50), ms),
        "ttft_ms_p99": _scale(percentile(ttft, 99), ms),
        "tpot_ms_p50": _scale(percentile(tpot, 50), ms),
        "tpot_ms_p99": _scale(percentile(tpot, 99), ms),
        "e2e_s_p50": percentile(e2e, 50),
        "e2e_s_p99": percentile(e2e, 99),
    }


def _scale(value, factor):
    return None if value is None else value * factor


async def send_request(client, url: str, model: str, prompt: str, max_tokens: int, timeout_s: float) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,  # vLLM extension: always generate exactly max_tokens
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, chunks, usage = None, 0, None
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                return RequestResult(ok=False, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                event = parse_sse_line(line)
                if event is None:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    except Exception as exc:  # network errors, timeouts
        return RequestResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    e2e = time.perf_counter() - t0
    return RequestResult(
        ok=True,
        ttft_s=ttft,
        e2e_s=e2e,
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
        output_tokens=usage.get("completion_tokens", chunks) if usage else chunks,
    )


async def run_level(client, url, model, concurrency, num_requests, input_len, output_len, timeout_s, rng):
    sem = asyncio.Semaphore(concurrency)
    prompts = [make_prompt(input_len, rng) for _ in range(num_requests)]

    async def bounded(prompt):
        async with sem:
            result = await send_request(client, url, model, prompt, output_len, timeout_s)
            result.concurrency = concurrency
            return result

    t0 = time.perf_counter()
    results = await asyncio.gather(*(bounded(p) for p in prompts))
    return list(results), time.perf_counter() - t0


async def wait_for_server(client, base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if (await client.get(f"{base_url}/health", timeout=5)).status_code == 200:
                return
        except Exception:
            pass
        print(f"waiting for {base_url}/health ...", flush=True)
        await asyncio.sleep(10)
    raise SystemExit(f"server at {base_url} not healthy after {timeout_s:.0f}s")


async def main_async(args) -> dict:
    import httpx

    rng = random.Random(args.seed)
    url = f"{args.base_url}/v1/completions"
    levels = [int(c) for c in args.concurrency.split(",")]
    os.makedirs(args.output, exist_ok=True)
    limits = httpx.Limits(max_connections=max(levels) + 8, max_keepalive_connections=max(levels) + 8)

    async with httpx.AsyncClient(limits=limits) as client:
        await wait_for_server(client, args.base_url, args.server_timeout)
        print(f"warmup: {args.warmup_requests} requests", flush=True)
        await run_level(client, url, args.model, min(4, args.warmup_requests), args.warmup_requests,
                        args.input_len, args.output_len, args.request_timeout, rng)

        summaries, all_results = [], []
        for c in levels:
            n = max(args.min_requests, c * args.requests_per_concurrency)
            results, wall = await run_level(client, url, args.model, c, n, args.input_len, args.output_len,
                                            args.request_timeout, rng)
            summary = summarize(results, c, wall)
            summaries.append(summary)
            all_results.extend(results)
            print(f"concurrency={c:>4}  out_tok/s={summary['output_tokens_per_s']:>9.1f}  "
                  f"ttft_p99={summary['ttft_ms_p99'] or 0:>8.1f}ms  e2e_p99={summary['e2e_s_p99'] or 0:>6.2f}s  "
                  f"failed={summary['failed']}", flush=True)
            errors = [r.error for r in results if r.error]
            if errors:
                print(f"  first error: {errors[0]}", flush=True)

    report = {
        "kind": "inference",
        "run_id": args.run_id,
        "accelerator_profile": args.accelerator_profile,
        "server": args.server_label,
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "levels": summaries,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output, "requests.csv"), "w", newline="") as f:
        fields = ["concurrency", "ok", "ttft_s", "tpot_s", "e2e_s", "prompt_tokens", "output_tokens", "error"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            row = asdict(r)
            row["tpot_s"] = r.tpot_s
            w.writerow({k: row[k] for k in fields})
    print(f"results written to {args.output}", flush=True)
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True, help="Model name exactly as the server registered it.")
    p.add_argument("--accelerator-profile", required=True)
    p.add_argument("--server-label", default="vllm")
    p.add_argument("--run-id", default=os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--concurrency", default="1,4,16,32,64")
    p.add_argument("--requests-per-concurrency", type=int, default=4)
    p.add_argument("--min-requests", type=int, default=32)
    p.add_argument("--warmup-requests", type=int, default=8)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None):
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
