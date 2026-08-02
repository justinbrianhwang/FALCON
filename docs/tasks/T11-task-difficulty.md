# Task T11 — Fix accuracy saturation in the synthetic task

**Assignee:** Kimi
**Problem:** `experiments/run_synthetic.py` reports accuracy=1.0000 from round 0 — clusters are
linearly separable, so failures produce no measurable accuracy gap (Plan §17.3 explicitly forbids
"no measurable failure" severities). Loss still moves, but the primary benchmark metric is dead.

**Schema (PM already added):** `DatasetConfig.class_separation: float = 1.0` (cluster-mean
distance relative to noise; lower = harder) and `label_noise: float = 0.0` (fraction of TRAIN
labels flipped, deterministic from `DatasetConfig.seed`; eval labels never flipped).

## Deliverables

1. `falcon/pipeline/synthetic_data.py`: scale cluster-mean separation by `class_separation`;
   apply `label_noise` to training partitions only, drawn from the partition generator.
2. **Calibrate the reference configs** (`configs/cases/*.yaml`, `configs/experiments/*.yaml`):
   choose `class_separation`/`label_noise` so the CLEAN reference run lands at final accuracy
   ≈ 0.85–0.92 with a visibly increasing accuracy curve (not saturated by round 0). Document the
   chosen values and the resulting curve in the YAML comments.
3. Re-verify each of the four failure cases still produces a measurable ACCURACY gap vs its
   reference (not just loss), adjusting failure severities if needed — record the gaps in
   tests/integration/test_failure_runs.py assertions.
4. Regenerate `tests/fixtures/golden_stage_hashes.json` for the new reference config (hashes
   change). Update any test that pinned saturated-accuracy expectations (e.g. `> 0.9` guards,
   E1 smoke). E0/E1 smoke must still pass.
5. `experiments/run_synthetic.py` output should now show a non-trivial accuracy trajectory.

Rules: you own falcon/pipeline, configs, the affected tests, and fixtures. Do not touch
falcon/schema (already done), matcher/metrics/attribution/reporting/intervention internals.
Full suite green (`python -m pytest tests -q`). No git commit.
