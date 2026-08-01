# Task T9 — E1 observational-equivalence experiment harness

**Assignee:** Kimi
**Contract:** Plan.md §20 E1, §35 Tasks 6–7. You own `experiments/` and `falcon/experiments/` (new package if needed). Consume public APIs only; modify no other package.

## Goal (harness only — paper-scale execution comes later)

Build `experiments/e1_equivalence.py` that, given a YAML experiment spec:

1. **Pair construction:** two failure cases with DIFFERENT origin stages (default: selection/minority_exclusion vs compression/aggressive_topk) on the same reference config.
2. **Severity matching:** bisection on each failure's severity parameter so both produce a terminal primary-metric gap within `gap_tolerance` of a target gap (e.g. 0.05 ± 0.01). Cap iterations; record the search trace. Refuse (clear message) if unmatchable within bounds.
3. **Three localization rules** on each matched pair (Plan §35 Task 7):
   - terminal-only: `NearestCentroidStageClassifier` on terminal features (fit on a small labeled set of runs generated across the four failure types — document the protocol);
   - passive anomaly: `passive_localize`;
   - FALCON: `analyze_pair` origin (respecting `outcome` — unresolved is a valid answer, never coerce).
4. **Output:** `results/e1/<case_id>/` — matched configs, severity traces, per-rule predictions vs ground truth, and a summary JSON + small markdown table. Everything reproducible from the YAML + seed.

Also add `configs/experiments/e1_smoke.yaml` — a tiny smoke config (few clients/rounds) and an integration test `tests/integration/test_e1_smoke.py` that runs the whole harness end to end on it (fast) and asserts the output files exist and FALCON's prediction for the smoke pair matches ground truth OR is explicitly `unresolved`.

Rules: numpy/pydantic/pyyaml/stdlib only; deterministic given the YAML seed; full suite green; no git commit.
