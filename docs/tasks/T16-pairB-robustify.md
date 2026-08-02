# Task T16 — Robust pair-B matching + metric-disagreement documentation

**Assignee:** Codex
**Context:** E2 pair B (local vs aggregation, configs/experiments/main/e2_local_vs_aggregation.yaml)
fails gap-matching at most seeds because `wrong_sample_weights/corrupted` has a SIGN-UNSTABLE
effect on LOSS (measured: gaps −0.021…+0.008 across seeds 1–5, non-monotone in intensity), while
its ACCURACY effect is strong and monotone (+0.038/+0.070/+0.118 at intensity 0.5/1/2, seed 42,
2 clients/round). PM verdict: this is Plan §14.10 metric disagreement observed in the wild —
document it, don't tune it away.

## Deliverables

1. `experiments/e1_equivalence.py`: support `metric: accuracy` matching robustly (accuracy is
   step-quantized at 1/eval_n — bisection must treat within-half-step as converged rather than
   oscillate). Keep loss matching unchanged.
2. Update `configs/experiments/main/e2_local_vs_aggregation.yaml` to match on
   `metric: accuracy, higher_is_better: true` with an attainable target band (measure, then pin;
   document measured per-seed endpoint gaps in YAML comments).
3. Verify seeds 1–5 all match and run end-to-end; report the Top-1 table for pair B in your
   summary (do not fabricate — if a seed is genuinely unmatchable, say so and document why).
4. New note `docs/notes/metric-disagreement-aggregation.md`: the corrupted-weights loss-vs-
   accuracy sign disagreement with the measured per-seed numbers from your runs (get them from
   the severity search traces), positioned as Plan §14.10/§21.5 evidence: "stage attribution and
   even failure DIRECTION depend on the outcome metric".
5. Regression test: accuracy-matching path in the harness (tiny config, converges, no oscillation).

Rules: you own experiments/e1_equivalence.py, that YAML, the note, and tests for the harness.
Full suite green. No git commit.
