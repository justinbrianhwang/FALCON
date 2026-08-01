# Task T6 — Attribution metrics + analyzer v0

**Assignee:** Codex
**Contract:** docs/CONTRACTS.md; Plan.md §14.1–14.5, §15.1–15.3. You own `falcon/metrics/` and `falcon/attribution/`; read-only elsewhere. Pure functions over schema objects — no pipeline execution, no I/O beyond what tests build.

## Deliverables

### 1. `falcon/metrics/effects.py`

All metric-direction-aware: `higher_is_better: bool` parameter, internally normalize so larger = better.

- `failure_gap(m_ref: float, m_fail: float, higher_is_better=True) -> float` — Plan §14.1 `G`.
- `sre(m_restored, m_fail, higher_is_better) -> float`; `nsre(m_restored, m_ref, m_fail, ...)` — §14.2. `nsre` must REFUSE (return `None`) when `|G| < min_gap` (parameter, default 1e-9 — caller passes the predeclared τ).
- `sie(m_ref, m_injected, ...)`, `nsie(...)` — §14.3, same guard.
- `bis(nsre, nsie, lam=0.5) -> float | None` — §14.4; None if either input is None.
- `sham_adjusted(effect, sham_effect) -> float` — §14.5.

### 2. `falcon/attribution/analyzer.py`

`attribute(pair: PairValidationReport, interventions: list[InterventionResult], *, metric: str, m_ref: float, m_fail: float, higher_is_better=True, min_gap: float, sham_tolerance: float) -> AttributionReport`

Logic (Plan §15.2 provisional origin criteria, §15.3 ambiguity):

- Group interventions by stage; use restore/inject/sham results for `metric` (final-round value from `outcome_metrics`).
- Invalid interventions (valid=False) are excluded and named in `notes`.
- Sham gate: if any stage's |sham deviation| > sham_tolerance → the WHOLE report is flagged: `notes` gets "SHAM_VIOLATION:<stage>" and origin_ranking is emptied (a method reporting large sham effects is invalid, Plan §12.4).
- Per-stage score: `bis` when both nSRE/nSIE exist, else the one that exists, else stage excluded.
- `origin_ranking`: stages ordered by (earliest first_divergence evidence, then score). Earliest-decisive-stage rule §15.2: if the pair's `first_divergence_stage` has score >= (max score - epsilon_tie), it ranks first even when a downstream stage scores higher; the downstream stage gets role "carrier_or_amplifier" in `roles`.
- Roles dict (provisional, §14.9 candidate logic): origin candidate = ranked first; stages with score < bystander_threshold (parameter, default 0.1) → "bystander"; negative score → "suppressor_candidate"; else "carrier_or_amplifier".
- Ambiguity: if |G| < min_gap → status note "INSUFFICIENT_FAILURE_GAP", empty ranking. If top two scores within epsilon_tie → note "UNRESOLVED_BETWEEN:<s1>,<s2>" but keep both in order.
- `failure_gap` dict in report: `{metric: G}`. `stage_effects`: per stage `{"SRE","SIE","nSRE","nSIE","BIS","sham_dev"}` (None → omit key).

### 3. Tests: `tests/unit/test_effects.py`, `tests/unit/test_analyzer.py`

- Hand-computed numeric cases for every §14 formula, both metric directions, the nSRE>1 and negative cases from §14.2's interpretation table, min_gap refusal.
- Analyzer: synthetic InterventionResult sets covering — clean single-stage attribution; downstream-restoration trap (downstream scores higher but first_divergence stage wins, downstream labeled carrier); sham violation kills the report; insufficient gap; tie → UNRESOLVED; invalid interventions excluded and noted.
- Build inputs directly as pydantic objects — no pipeline runs.

## Rules

numpy/pydantic/stdlib only. Do not touch falcon/pipeline, falcon/intervention (Kimi works there concurrently), or their tests. `python -m pytest tests/unit/test_effects.py tests/unit/test_analyzer.py -q` green. No git commit.
