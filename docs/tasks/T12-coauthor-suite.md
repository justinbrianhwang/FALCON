# Task T12 — Co-author supplementary experiment suite

**Assignee:** Codex — START ONLY AFTER T11 (task-difficulty calibration) lands, its YAML values feed this.
**Goal:** the co-author machine must produce COMPLEMENTARY evidence, not duplicate runs.

## Division of labor (PM decision)

| Machine | Experiments |
|---|---|
| Main (user) | E1 core matrix (primary seeds), later Tier 1 CIFAR |
| Co-author | (a) cross-machine determinism, (b) seed replication, (c) heterogeneity sweep |

Scientific value per §12.3/§21: (a) decides whether the "bitwise" replay-level claim holds across
OS/hardware — impossible to test on one machine; (b) independent seeds double the seed count for
§21.2 CIs without re-running ours; (c) fills the Dirichlet/heterogeneity axis of E3-lite.

## Deliverables

1. `configs/experiments/coauthor/` —
   - `e0_crossmachine.yaml`: EXACTLY the calibrated main reference configs (same seeds!) so their
     recorded stage hashes are directly diffable against `tests/fixtures/golden_stage_hashes.json`;
   - `e1_seeds.yaml`: the E1 experiment with a DISJOINT seed list (document: main = seeds 1–5,
     co-author = seeds 101–105);
   - `e1_heterogeneity.yaml`: E1 at heterogeneity levels the main machine does NOT run
     (document the split in the YAML header).
2. `experiments/run_coauthor_suite.py` — ONE command: runs E0 cross-machine, E1 seeds, E1
   heterogeneity in sequence, writes everything under `results/coauthor/`, then invokes
   `scripts/collect_output.py` so the run ends with a ready-to-send `tmp/Output_<timestamp>.zip`.
   Print a clear final line telling the co-author which file to send. Failures in one experiment
   must not abort the rest (collect partial results + mark FAILED in a summary).
3. `scripts/compare_crossmachine.py` — OUR side: takes an Output zip path, extracts their E0
   stage hashes, diffs against the local golden fixture, prints per-boundary match/mismatch and
   a verdict (`bitwise-portable` / `machine-dependent`).
4. Update `docs/COAUTHOR.md` section 3: replace the generic run commands with
   `python experiments/run_coauthor_suite.py` and one sentence on what it does.
5. Smoke test `tests/integration/test_coauthor_suite_smoke.py` (tiny overrides, fast) asserting
   the suite writes its summary and the collector zip.

Rules: consume public APIs; you own the new files + docs/COAUTHOR.md §3 edit; full suite green; no git commit.
