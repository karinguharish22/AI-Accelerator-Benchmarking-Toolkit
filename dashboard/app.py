"""AccelBench advisor: pick the cheapest accelerator that meets a serving SLA.

streamlit run dashboard/app.py -- --results results
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from analysis.costs import (  # noqa: E402
    PRICING_MODES,
    best_per_profile,
    inference_rows,
    is_synthetic,
    load_inference_runs,
    load_pricing,
    load_training_runs,
    recommend,
    training_rows,
)


def cli_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=os.environ.get("ACCELBENCH_RESULTS", "results"))
    p.add_argument("--pricing", default="configs/pricing.yaml")
    return p.parse_known_args()[0]


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, log_x: bool = True):
    scale = alt.Scale(type="log", base=2) if log_x else alt.Scale()
    return (
        alt.Chart(df, title=title)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, scale=scale),
            y=alt.Y(y),
            color="accelerator_profile:N",
            tooltip=["accelerator_profile", "concurrency", x, y],
        )
        .properties(height=300)
    )


def main():
    args = cli_args()
    st.set_page_config(page_title="AccelBench", layout="wide")
    st.title("AccelBench: accelerator cost-performance advisor")

    with st.sidebar:
        results_dir = st.text_input("Results folder", args.results)
        pricing_path = st.text_input("Pricing file", args.pricing)
        mode = st.radio("Pricing", PRICING_MODES, format_func=lambda m: m.replace("_", "-"))
        st.caption("Prices in the pricing file are placeholders until you update them.")

    pricing = load_pricing(pricing_path)
    train_runs = load_training_runs(results_dir)
    infer_runs = load_inference_runs(results_dir)
    if not train_runs and not infer_runs:
        st.info(f"No results in `{results_dir}`. Run the benchmarks and `make fetch-results`, "
                "or try `python -m analysis.make_demo_data` for synthetic sample data.")
        return
    if is_synthetic(train_runs, infer_runs):
        st.error("SYNTHETIC DEMO DATA: these numbers are generated, not measured. Do not report them.")

    infer = pd.DataFrame(inference_rows(infer_runs, pricing, mode))
    train = pd.DataFrame(training_rows(train_runs, pricing, mode))
    tab_rec, tab_inf, tab_train = st.tabs(["Recommendation", "Serving", "Fine-tuning"])

    with tab_rec:
        if infer.empty:
            st.info("No serving results yet.")
        else:
            c1, c2, c3 = st.columns(3)
            max_ttft = c1.number_input("Max p99 time to first token (ms)", value=1000.0, step=100.0)
            max_tpot = c2.number_input("Max p99 time per output token (ms)", value=100.0, step=10.0)
            min_tps = c3.number_input("Min output tokens/s per replica", value=0.0, step=100.0)
            feasible = recommend(infer.to_dict("records"), max_ttft, max_tpot, min_tps or None)
            if not feasible:
                st.warning("No measured configuration meets these constraints.")
            else:
                top = feasible[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recommended", top["accelerator_profile"])
                m2.metric("Concurrency", int(top["concurrency"]))
                m3.metric("Output tokens/s", f"{top['output_tokens_per_s']:,.0f}")
                m4.metric("$ per 1M output tokens", f"{top['usd_per_million_output_tokens']:.3f}")
                st.subheader("Best operating point per accelerator")
                st.dataframe(pd.DataFrame(best_per_profile(feasible))[[
                    "accelerator_profile", "concurrency", "output_tokens_per_s", "ttft_ms_p99", "tpot_ms_p99",
                    "usd_per_million_output_tokens",
                ]], hide_index=True, use_container_width=True)

    with tab_inf:
        if infer.empty:
            st.info("No serving results yet.")
        else:
            a, b = st.columns(2)
            a.altair_chart(line_chart(infer, "concurrency", "output_tokens_per_s", "Throughput"),
                           use_container_width=True)
            b.altair_chart(line_chart(infer, "concurrency", "usd_per_million_output_tokens", "$ per 1M output tokens"),
                           use_container_width=True)
            a.altair_chart(line_chart(infer, "concurrency", "ttft_ms_p99", "p99 TTFT (ms)"), use_container_width=True)
            b.altair_chart(line_chart(infer, "output_tokens_per_s", "tpot_ms_p99", "Latency-throughput frontier",
                                      log_x=False), use_container_width=True)
            st.dataframe(infer, hide_index=True, use_container_width=True)

    with tab_train:
        if train.empty:
            st.info("No fine-tuning results yet.")
        else:
            train["config"] = train["accelerator_profile"] + " / " + train["padding"]
            cols = st.columns(3)
            for col, metric, title in zip(cols, ["tokens_per_s", "usd_per_million_tokens", "mfu"],
                                          ["Tokens/s", "$ per 1M training tokens", "MFU"], strict=True):
                col.altair_chart(
                    alt.Chart(train, title=title).mark_bar().encode(
                        x=alt.X("config:N", sort=None), y=metric, color="accelerator_profile:N",
                        tooltip=["config", metric]),
                    use_container_width=True,
                )
            st.dataframe(train, hide_index=True, use_container_width=True)


main()
