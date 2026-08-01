# Task T4/T5/T8-F — Fix round-2 adversarial review findings

**Assignee:** Kimi (original author)
**Input:** docs/reviews/T4T5T8-review-by-codex.md — PM verdicts + design decisions below.
**Contract change:** docs/CONTRACTS.md §3 is amended to v0.2 (round-keyed client streams) — implement it.

| # | Severity | PM verdict / decision | Action |
|---|----------|----------------------|--------|
| 1 | CRITICAL | Confirmed. **Contract amended.** | Switch local training to stream `client.<id>.round.<t>.dataloader`. Update ClientLocalState.rng_state key accordingly. Add Codex's aggregate-restoring selection-overlay counterexample as a regression test (client_1's round-1 update must now be IDENTICAL). Note: this changes all recorded hashes — update any golden expectations. |
| 2 | CRITICAL | Confirmed. **Sham redesigned.** | Sham = (a) run a NO-overlay replay of the target; require every recorded boundary hash to match the target's recording — any mismatch → `valid=False, reason="replay_drift:<round>/<stage>"`; (b) round-trip the LIVE boundary state through Recorder save/load in a temp dir and overlay that (serialization test), never the recorded target state; (c) evaluation-stage sham compares recomputed outcome, not self-replaced outcome. Add the drifted-metadata fixture from the review — sham must REJECT it. |
| 3 | MAJOR | Confirmed. **Reject, don't warn.** | Lineage (base_model_hash / uncompressed_hash) mismatch → `valid=False, reason="lineage_mismatch"`. Remove `warning_base_model_mismatch`. A future `raw_delta_transplant` mode can be added when scientifically justified — not now. Update the test that institutionalized the warning path. |
| 4 | MAJOR | Confirmed | Engine loads BOTH runs' RunMetadata and requires: same seed, equal dataset config, config delta limited to `failure` — else `valid=False, reason="incompatible_runs"`. Keep shape/dtype/finiteness checks per stage (parameterize tests over local/compression/aggregation, both directions). |
| 5 | MAJOR | Confirmed | Scope validation: must be absent or a nonempty list of unique nonempty strings (else invalid); `round_id` range-checked before loading; recorder/pydantic/path errors translated to `valid=False` stable reasons; overlay must fire exactly once (assert internally). Tests for None/empty/duplicate/string scope, corrupt hash, wrong state type, out-of-range round. |
| 6 | MAJOR | Confirmed | Reject non-finite `lr_multiplier` (and any injector float param) at construction. Passive baseline: non-finite stage score → return it as `float("nan")` but `passive_localize` must raise `ValueError("non-finite anomaly scores")` — never argmax over NaN; same guard in `NearestCentroidStageClassifier.predict`. Tests for NaN/inf in `_relative_l2`, localize, classifier, injector params. |
| 7 | MAJOR | Confirmed. **Undersubscription allowed.** | `select_clients` selects `min(clients_per_round, len(pool))`; SelectionState records what happened (selected fewer). Runner must not crash; integration test where exclusion leaves pool < clients_per_round and the run completes with a recorded undersubscribed round. |
| 8 | MAJOR | Confirmed | Commit a golden boundary-hash fixture (JSON of stage_hashes for the reference config, generated once at this fix's HEAD) and assert the no-overlay runner reproduces it; add finding-1 and finding-2 regression tests; parameterize validation tests per stage/direction; pin one exact RNG draw expectation in failure-strength tests. |

Order: 1 → 7 → 2 → 3 → 4 → 5 → 6 → 8 (1 and 7 change hashes; golden fixture last).

Rules: you own falcon/pipeline, falcon/failures, falcon/intervention, falcon/baselines and their tests. Do not touch falcon/schema, falcon/replay, falcon/recorder, falcon/matcher, falcon/metrics, falcon/attribution, falcon/reporting (Codex will be fixing those concurrently — if the reporting end-to-end test breaks because sham now correctly rejects, note it in your summary; do not edit it). Full suite green except possibly that file. No git commit.
