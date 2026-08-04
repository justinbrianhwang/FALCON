# T22 — E5 aggregator-role matrix (H4)

Owner: Kimi. New files only — do NOT modify anything under `falcon/`.

## Goal

Test hypothesis H4 (Plan §8): the aggregation stage's propagation ROLE depends on the
aggregation rule. Under `weighted_mean`, every E1 run showed aggregation as a faithful
carrier (`CARRIER_TIE_RESOLVED:aggregation` in FALCON attributions). Robust rules
(`median`, `trimmed_mean` — already implemented in `falcon/pipeline/stages.py::aggregate`,
tested in `tests/unit/test_stages.py`) may instead SUPPRESS upstream damage, changing
both the failure gap and the tie structure.

## Deliverable

`experiments/run_e5_aggregators.py` with `--smoke` (2 rules x 1 failure, 4 rounds) and
full mode. Pattern-match `experiments/run_coauthor_cifar.py` (same _record/analyze_pair
flow — read it first).

Matrix (full mode): aggregation rule in
  {weighted_mean, median, trimmed_mean (parameters: {"beta": 0.2})}
x failure in
  {selection/minority_exclusion sev 2, local/lr_misconfig sev 2, compression/aggressive_topk sev 2}
= 3 rules x 3 failures = 9 pairs (one reference run per rule, run_id `ref_<rule>`).

Base config: load `configs/cases/synthetic_reference.yaml`, override
`aggregation.rule` (and parameters for trimmed_mean). Failure specs: copy the
parameter shapes from `configs/cases/synthetic_selection_failure.yaml`,
`synthetic_local_failure.yaml`, `synthetic_compression_failure.yaml` (read them; keep
their calibrated parameters, severity as listed there, active_rounds as in those files).

Per pair: `analyze_pair(out_dir, f"ref_{rule}", f"fail_{rule}_{failure}",
metric="accuracy", higher_is_better=True, min_gap=0.005, sham_tolerance=1e-9)`,
write `render_markdown` report + a summary row.

## Output

`results/e5_aggregators/summary.json` + `e5_table.md`: rows = failure, cols = rule,
cell = outcome / top-1 prediction / failure gap / aggregation-stage role. The role
comes from `report.roles.get("aggregation")` and whether notes contain
`CARRIER_TIE_RESOLVED:aggregation`; also record aggregation's nSRE/nSIE from
`report.stage_effects` and flag negative scores as suppressor evidence.

## Acceptance

- `python experiments/run_e5_aggregators.py --smoke` exits 0 and writes the table.
- Full mode runs in minutes (synthetic Tier-0, 20 rounds, 10 clients).
- No edits outside `experiments/run_e5_aggregators.py`.
- cp949-safe stdout (ASCII only in prints).
