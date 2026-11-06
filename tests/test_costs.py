import pytest

from analysis.costs import best_per_profile, cost_per_million_tokens, inference_rows, recommend, training_rows

PRICING = {
    "gpu": {"chips": 1, "peak_bf16_tflops_per_chip": 100, "hourly_usd": {"spot": 1.0, "on_demand": 3.0}},
    "tpu": {"chips": 1, "peak_bf16_tflops_per_chip": 200, "hourly_usd": {"spot": 2.0, "on_demand": 4.0}},
}


def test_cost_per_million_tokens():
    # $3.60/h at 1000 tok/s -> 3.6M tokens/h -> $1 per 1M tokens.
    assert cost_per_million_tokens(3.6, 1000) == pytest.approx(1.0)
    assert cost_per_million_tokens(None, 1000) is None
    assert cost_per_million_tokens(1.0, 0) is None


def _level(c, tps, ttft, tpot, failed=0):
    return {"concurrency": c, "requests": 32, "succeeded": 32 - failed, "failed": failed, "wall_s": 10,
            "output_tokens_per_s": tps, "ttft_ms_p99": ttft, "tpot_ms_p99": tpot}


def _inference_runs():
    return [
        {"run_id": "r", "accelerator_profile": "gpu", "model": "m", "input_len": 512, "output_len": 256,
         "levels": [_level(1, 100, 50, 20), _level(32, 1000, 900, 60)]},
        {"run_id": "r", "accelerator_profile": "tpu", "model": "m", "input_len": 512, "output_len": 256,
         "levels": [_level(1, 150, 60, 15), _level(32, 3000, 1500, 50)]},
    ]


def test_recommend_respects_sla_and_sorts_by_cost():
    rows = inference_rows(_inference_runs(), PRICING, "spot")
    # gpu@32: $1/h / 1000 tok/s = $0.278/1M; tpu@1: $2/h / 150 tok/s = $3.70/1M; tpu@32 violates TTFT.
    feasible = recommend(rows, max_ttft_p99_ms=1000, max_tpot_p99_ms=100)
    assert [(r["accelerator_profile"], r["concurrency"]) for r in feasible][0] == ("gpu", 32)
    assert ("tpu", 32) not in [(r["accelerator_profile"], r["concurrency"]) for r in feasible]

    # Loosen the SLA and the TPU's high-concurrency point becomes the cheapest.
    loose = recommend(rows, max_ttft_p99_ms=2000)
    assert (loose[0]["accelerator_profile"], loose[0]["concurrency"]) == ("tpu", 32)
    assert [r["accelerator_profile"] for r in best_per_profile(loose)] == ["tpu", "gpu"]


def test_recommend_skips_levels_with_failures():
    runs = _inference_runs()
    runs[0]["levels"][1]["failed"] = 3
    feasible = recommend(inference_rows(runs, PRICING, "spot"), max_ttft_p99_ms=1000)
    assert ("gpu", 32) not in [(r["accelerator_profile"], r["concurrency"]) for r in feasible]


def test_training_rows_cost_and_mfu():
    run = {"run_id": "r", "accelerator_profile": "gpu", "model": "m", "num_devices": 1, "seq_len": 1024,
           "padding": "max_length", "global_batch_size": 4, "tokens_per_s": 1000, "tokens_per_s_per_device": 1000,
           "step_time_s": 4.0, "flops_per_token": 1e10, "total_wall_s": 3600}
    (row,) = training_rows([run], PRICING, "on_demand")
    assert row["usd_per_million_tokens"] == pytest.approx(3.0 / 3.6)
    assert row["usd_per_run"] == pytest.approx(3.0)
    assert row["mfu"] == pytest.approx(0.1)  # 1e13 FLOP/s of 1e14 peak
