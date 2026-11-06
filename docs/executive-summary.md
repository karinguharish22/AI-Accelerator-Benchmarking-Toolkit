# Executive summary deck: outline

A 10-slide structure for presenting the results to a technical buyer or an executive.
Fill it in from `results/report/report.md`. Use only measured numbers, and state the
pricing date.

1. **The question.** "What is the most cost-effective accelerator to fine-tune and serve our
   LLM at our latency target?" Add one line of business context, such as the expected
   tokens per month.
2. **Answer first.** The recommended configuration, its cost per 1M tokens, and the savings
   against the alternative, as one headline number.
3. **What we tested.** Model, dataset, accelerators, and the fact that the code and settings
   were identical on each. Include the architecture diagram.
4. **Fine-tuning results.** Tokens/s per chip, MFU and $/1M tokens for each accelerator, as a
   bar chart (`training.png`).
5. **Serving results.** Cost vs concurrency (`inference_cost.png`) with the SLA line marked.
6. **Latency.** The latency-throughput frontier (`inference_frontier.png`): where each
   option breaks the SLA.
7. **What we optimized.** One or two profiling findings, with before and after numbers
   (for example, removing XLA recompilation).
8. **Migration effort.** What changed in the code (the `Backend` table from the migration
   playbook), risks, and the validation done.
9. **Recommendation and rollout.** Phases, capacity strategy (Spot / reservation /
   flex-start), and the rollback plan.
10. **Next steps.** Pilot scope, success criteria, owners and timeline.

Tips: lead with the business number, keep one message per slide, put the methodology in the
appendix, and state the limits of the test openly.
