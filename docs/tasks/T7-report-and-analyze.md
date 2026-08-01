# Task T7 — Report generator + end-to-end analyze driver

**Assignee:** Codex
**Contract:** docs/CONTRACTS.md; Plan.md §11.7, §15.4, Appendix B. You own `falcon/reporting/`; call public APIs of matcher/intervention/attribution/metrics read-only. Do not modify other packages.

## Deliverables

### 1. `falcon/reporting/analyze.py`

`analyze_pair(runs_root: Path, reference_run_id: str, failure_run_id: str, *, metric: str, higher_is_better: bool, min_gap: float, sham_tolerance: float, rounds: list[int] | None = None) -> tuple[AttributionReport, list[InterventionResult]]`

Orchestration:

1. `validate_pair` — status INVALID_PAIR → return a report whose notes say so; run NO interventions (invalid pairs must not produce causal estimates, Plan §11.3).
2. Choose intervention round: `rounds` if given, else the pair's `first_divergence_round` (fallback: failure spec's `active_rounds[0]`).
3. For each of the four intervenable stages (selection, local, compression, aggregation) at each chosen round: restore (target=failure, source=reference), inject (target=reference, source=failure), sham (target=failure).
4. Pull `m_ref` / `m_fail` from the two runs' recorded final `OutcomeState.metrics[metric]`.
5. Feed everything to `attribute(...)` → AttributionReport.

### 2. `falcon/reporting/report.py`

`render_markdown(report: AttributionReport, interventions: list[InterventionResult], *, ground_truth: FailureSpecification | None) -> str`

Sections (Plan §11.7): pair validity; terminal failure summary (metric, G); intervention effect table (stage x SRE/SIE/nSRE/nSIE/BIS/sham_dev); origin ranking + roles; §15.4-style counterfactual explanation sentence generated from measured values ("Restoring <stage> closes X% of the gap; injecting reproduces Y%"); warnings/assumptions; and — benchmark mode only — a clearly separated "Ground truth (benchmark)" section comparing predicted origin vs injected stage. Measured evidence, inferred role, and ground truth must be visually separate.

### 3. CLI: `falcon/reporting/__main__.py`

`python -m falcon.reporting --runs-root runs --reference ref_001 --failure fail_001 --metric accuracy [--higher-is-better/--lower-is-better] [--min-gap 0.005] [--sham-tolerance 1e-9] [--output report.md] [--json report.json]`

### 4. Tests: `tests/integration/test_analyze_end_to_end.py`

Build one reference + one T4 failure run (reuse a cheap config from existing integration tests), then `analyze_pair` end to end: origin ranking's first element == injected stage; sham deviations ~0; invalid pair (tampered seed) → no interventions, INVALID note; markdown renders with all sections and correct percentages.

## Rules

numpy/pydantic/stdlib only. `python -m pytest tests/integration/test_analyze_end_to_end.py -q` green plus full suite green. No git commit.
