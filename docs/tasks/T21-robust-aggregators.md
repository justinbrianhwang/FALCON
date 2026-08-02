# Task T21 — Robust aggregation rules: median, trimmed_mean

**Assignee:** Kimi
**Goal:** implement the two aggregation rules the schema already declares
(`AggregationConfig.rule`: "median", "trimmed_mean") — currently NotImplementedError in
`falcon/pipeline/stages.py`. These are the E5 (aggregator-role matrix) prerequisites: Plan §4.4,
§18.4.

## Deliverables

1. `stages.aggregate`:
   - `median`: coordinate-wise median over client updates (unweighted by definition — ignore
     weights, record the actually-used uniform weights in AggregationState.weights; document).
     Even-count median = lower-index... no: use numpy's standard midpoint average (document it).
   - `trimmed_mean`: coordinate-wise, trim fraction `beta` from each end
     (`parameters: {beta: 0.1}` default; validate 0 <= beta < 0.5; trims `floor(beta * n)`
     clients per side per coordinate), unweighted mean of the rest; document tie/ordering
     determinism (np.sort is stable on values — no client-order dependence).
   - Both dtype-preserving (float64 synthetic / float32 Tier-1), deterministic, no RNG.
2. Tests (`tests/unit/test_stages.py` additions): hand-computed median and trimmed_mean cases
   (odd/even client counts, beta edge values, validation failures), dtype preservation,
   determinism, and AggregationState fields (accepted/rejected: for trimmed_mean the
   per-coordinate trimming has no whole-client reject set — record all received as accepted and
   put beta in a documented place; state your choice in the summary).
3. One integration test: synthetic reference run with `rule: median` completes, is deterministic
   (two runs bit-identical), and clean-run accuracy stays within a sane band (measure, assert
   loosely).

Rules: you own falcon/pipeline/stages.py + tests. No schema changes (rules already declared).
Use ~/miniconda3/envs/falcon/python.exe (FALCON_DATA_ROOT inline if needed). Full suite green.
No git commit. Another developer is concurrently editing configs/experiments/main/ — don't touch it.
