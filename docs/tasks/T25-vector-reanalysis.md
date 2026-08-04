# T25 — Outcome-vector reanalysis of the recorded synthetic + MNIST matrix

Owner: Kimi. New file only: `experiments/run_vector_reanalysis.py`. Do NOT modify
anything under `falcon/`.

## Goal

Plan §14.10: extend the metric-disagreement evidence from CIFAR to every recorded
matched pair of the official matrix, using `falcon.reporting.analyze.analyze_pair_vector`
(one intervention replay per pair, re-attributed under every metric). Tier-0/MNIST
replays are cheap (seconds-minutes per pair).

## Input discovery

Walk `results/` for run roots that contain `runs/<ref_id>` and `runs/<fail_id>`
pairs (a run root is a directory with a `runs/` child). Start with these roots if
present, skip silently if absent:

- results/e1_main_seeds/** (synthetic pairs A and B, per seed)
- results/main_matrix/** (T19 layout)
- results/coauthor_cifar_smoke (MNIST smoke)

Read each root's runs/ subdirectories' metadata.json: a run with `failure: null`
(and empty `failures` if the field exists) is a reference; every other run in the
same root pairs with the reference that shares its `seed`. If a root has multiple
references with different seeds, match by seed. Skip roots where pairing is
ambiguous and record why in the summary (do not guess).

## Analysis

Import VECTOR from experiments/run_outcome_vector.py. For failure specs with a
`target_class` parameter, add `class_<target>_accuracy` like run_outcome_vector.py
does. min_gap=0.005 (loss 0.01 via VECTOR), sham_tolerance=1e-9. higher_is_better
comes from VECTOR.

Write `results/vector_reanalysis/summary.json` (one row per pair per metric:
root, ref_id, fail_id, ground-truth stage from metadata failure.stage, metric,
outcome, top-1, gap) and `results/vector_reanalysis/vector_matrix.md` — a table:
rows = (root, pair), cols = metrics, cells = outcome/top1-correctness/gap, plus a
final ACCURACY-OF-ATTRIBUTION line per metric (correct unique_origin count /
sufficient-gap count).

`--limit N` flag: analyze only the first N pairs (for smoke; default all).
`--roots a,b` flag: restrict discovery.

## Acceptance

- `python experiments/run_vector_reanalysis.py --limit 2` exits 0 on this machine.
- No edits outside the new script. cp949-safe stdout (ASCII only).
