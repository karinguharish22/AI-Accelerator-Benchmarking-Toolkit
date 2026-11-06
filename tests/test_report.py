import os

from analysis import make_demo_data
from analysis.report import build_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_data_end_to_end(tmp_path):
    results = tmp_path / "results"
    make_demo_data.main(["--out", str(results)])
    out = tmp_path / "report"
    report = build_report(str(results), os.path.join(ROOT, "configs", "pricing.yaml"), "spot", str(out),
                          max_ttft_p99_ms=1000, max_tpot_p99_ms=100)

    assert "SYNTHETIC DEMO DATA" in report
    assert "## Fine-tuning (LoRA)" in report and "## Serving" in report
    assert "## Recommendation" in report
    for name in ["report.md", "training.csv", "inference.csv", "training.png", "inference_throughput.png",
                 "inference_cost.png", "inference_frontier.png"]:
        assert (out / name).exists(), name
