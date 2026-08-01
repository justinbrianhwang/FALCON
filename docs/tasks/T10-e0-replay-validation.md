# Task T10 — E0 replay-validation experiment harness

**Assignee:** Codex
**Contract:** Plan.md §20 E0, §12.3. You own `experiments/e0_replay_validation.py` and its test; consume public APIs only.

## Goal (harness only)

`experiments/e0_replay_validation.py --config configs/experiments/e0_smoke.yaml --output results/e0/`:

1. For each config in the YAML list: execute the same clean run twice (fresh Recorder each), compare ALL stage hashes; report agreement per boundary.
2. Sham battery: for each stage at a mid round, run a sham intervention; record deviations.
3. Checkpoint-restore equivalence (Plan §12.3 item 6): re-run with a mid-run recorded rng_state restored — verify the suffix matches the original run's hashes (use existing public APIs; if an API gap blocks this, implement the check as far as possible and document the gap in the output).
4. Output: `results/e0/report.json` (+ short markdown): per-config PASS/FAIL, mismatched boundaries if any, max |sham deviation|, and the declared replay level (`bitwise` if all hashes agree, else `mismatch`).

Exit code 1 if any config FAILs — this doubles as Gate G1 evidence.

Add `configs/experiments/e0_smoke.yaml` (tiny configs, both with and without a failure spec) and `tests/integration/test_e0_smoke.py` asserting the smoke run passes with `bitwise` level and near-zero sham deviations.

Rules: numpy/pydantic/pyyaml/stdlib only; full suite green; no git commit. The other developer is concurrently building experiments/e1_equivalence.py — avoid touching it.
