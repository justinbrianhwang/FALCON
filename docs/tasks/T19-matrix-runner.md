# Task T19 — Consolidated main-machine experiment matrix runner

**Assignee:** Codex
**Motivation:** pair A/B runs and seed replications were driven by ad-hoc inline scripts (PM's
shell). Plan §28 requires every reported number to come from a committed config + script.

## Deliverables

1. `experiments/run_main_matrix.py`: one command that runs the declared experiment matrix —
   `configs/experiments/main/matrix.yaml` listing entries `{name, config, seeds}` (initial
   content: pair A = e1_seeds.yaml seeds 1–5 and 101–105; pair B = e2_local_vs_aggregation.yaml
   seeds 1–5 and 101–105). Reuses the per-seed spec expansion pattern from run_coauthor_suite.py.
   Unmatchable seeds are recorded as EXCLUDED with the harness reason (Plan §21.6), never
   retried with a different band.
2. Aggregated outputs under `results/main_matrix/`: per-case rows (case, seed, ground truth,
   falcon/passive/terminal predictions, gap, notes) as `table1.csv` + `table1.md` with Top-1
   totals per method (matched cases only) and an exclusions table with reasons.
3. Smoke test `tests/integration/test_main_matrix_smoke.py` (tiny override matrix, fast).
4. Run the REAL matrix once at the end (takes a few minutes) and paste table1.md into your
   summary. Expected from prior runs: FALCON 40/40 matched, 3 exclusions (pair B seeds
   102/104/105) — report what you actually get.

Rules: you own the new files only. Full suite green. No git commit.
