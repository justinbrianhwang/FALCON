# Task T3/T6/T7-F — Fix round-2 adversarial review findings

**Assignee:** Codex (original author)
**Input:** docs/reviews/T3T6T7-review-by-kimi.md — PM verdicts + design decisions below.
**Schema change (already applied by PM):** `AttributionReport` gained `outcome: AttributionOutcome` and `origin_set: list[str]` (see falcon/schema/states.py). All analyzer/report code must fill them.

| # | Severity | PM verdict / decision | Action |
|---|----------|----------------------|--------|
| C1 | CRITICAL | Confirmed. **Two tolerances.** | Keep `epsilon_tie=1e-9` for float ties; add `decisive_margin: float = 0.05` used by the first-divergence promotion ("interventionally indistinguishable"). Thread through `attribute`, `analyze_pair`, and reporting CLI (`--decisive-margin`). Add the reviewer's upstream-tie regression test (promotion vs pipeline-order re-sort divergence case from M7). |
| C2 | CRITICAL | Confirmed. **Outcome field.** | Analyzer sets `outcome`: tie within margin → `unresolved` + `origin_set` = ALL tied stages (fixes minor 9), `origin_candidate` role suppressed; all-effects-negligible with first-divergence promotion → `unresolved`, never a confident origin; normal case → `unique_origin`. `render_markdown`: unresolved → print the tied set, no single-stage counterfactual sentence; benchmark verdict prints `unresolved`, not yes/no. `INSUFFICIENT_FAILURE_GAP` / `INVALID_PAIR` / sham violation map to their outcomes. |
| C3 | CRITICAL | Confirmed | Gap guard: `not isfinite(gap) or gap < min_gap` → outcome `insufficient_failure_gap` (note `NONPOSITIVE_FAILURE_GAP` when gap ≤ 0). Same finiteness+positivity in `effects.nsre/nsie` (return None). Reject non-finite `outcome_metrics[metric]` at grouping with an `INVALID_INTERVENTION` note. Tests: negative gap, NaN gap, NaN metric. |
| M1 | MAJOR | Confirmed | Orphan `.npz` without `.json` (or vice versa) at any boundary → `INVALID_PAIR` (fail closed). Prefer fingerprinting via `Recorder.load` + `hash_model` so recorder integrity checks apply. |
| M2 | MAJOR | Confirmed | Broaden the `hashes_loadable` guard to `Exception` (incl. `zipfile.BadZipFile`) → INVALID_PAIR, never a traceback. |
| M3 | MAJOR | Confirmed | Add `same_code_version` check → mismatch = warning (MATCHED_WITH_WARNINGS). |
| M4 | MAJOR | Confirmed | Empty pre-failure boundary set → warning "no pre-failure boundaries recorded — match rests on config/seed only" (forces MATCHED_WITH_WARNINGS). |
| M5 | MAJOR | Confirmed. **Mean over rounds.** | Group by `(stage, mode, round)`; per stage/mode aggregate = mean of per-round normalized effects; add `n_rounds` to stage_effects; note when rounds disagree in sign. |
| M6 | MAJOR | Confirmed | Ranked stage without a valid sham → note `SHAM_CONTROL_MISSING:<stage>`. No valid sham anywhere → outcome `unresolved` + note `NO_SHAM_CONTROLS`. Delete or use the dead `m_ref` sham baseline branch (PM: delete; reference-run shams can come later). |
| M7 | MAJOR | Confirmed | Add all six surviving-mutant killers as tests (bis penalty direction with nSRE<nSIE; negative gap; both matcher config checks; promotion-specific case; sham boundary `>=`). |
| minors | — | All confirmed | 1: matcher CLI exit 1 on INVALID_PAIR. 2: note `OVERSHOOT:<stage>` when nSRE or nSIE > 1 (§14.2 "requires analysis"). 3: comment BIS degeneracy at λ=0.5. 4: at equal magnitude prefer bidirectional evidence in ranking (document). 5: report `SAE` in stage_effects when sham exists. 6: docstring note that roles are provisional pre-§14.9. 7: warn when first divergence is outside the failure's active window. 8: analyze_pair degrades gracefully (missing evaluation/metric/corrupt state → notes + partial report, no traceback). 9: covered by C2 origin_set. |

**Coordination:** the other developer is concurrently changing `falcon/intervention/engine.py` sham semantics (docs/tasks/T4T5T8-fixes.md: sham may now return `valid=False, reason="replay_drift:..."`, lineage mismatch is now invalid, round-keyed RNG streams change recorded hashes). Your analyzer/report must treat those invalid shams via M6. Do not modify falcon/pipeline, falcon/intervention, falcon/failures, falcon/baselines or their tests. If `tests/integration/test_analyze_end_to_end.py` needs updating for the new semantics, that file is YOURS — update it.

Full suite green at the end (coordinate timing: run full pytest only after both fix tasks land if needed — note remaining failures explicitly in your summary). No git commit.
