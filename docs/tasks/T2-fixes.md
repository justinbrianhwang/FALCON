# Task T2-F — Fix findings from adversarial review

**Assignee:** Kimi (original T2 author)
**Input:** docs/reviews/T2-review-by-codex.md — PM verdicts below. Fix in this order.

| # | Severity | PM verdict | Action |
|---|----------|-----------|--------|
| 1 | CRITICAL | **Confirmed** (PM-verified at runner.py:57,65) | Record per-client stages ONCE per stage as a list: collect all `ClientLocalState` then `_record(recorder, round_id, "local", local_states)`; same for compression. |
| 2 | MAJOR | Confirmed | Reimplement minority partition to match documented semantics: designated subset = `max(1, round(num_clients * fraction))` clients; concentrate ~`concentration` share of ALL minority-class samples there; suppress minority label on non-designated clients. Add invariant tests (global rarity, concentration share, nonempty subset, determinism, float64). |
| 3 | MAJOR | Confirmed | After each consuming stage, snapshot the consumed stream's state into the returned state's `rng_state`: `SelectionState.rng_state = {"client_selection": <bit_generator.state>}`, `ClientLocalState.rng_state = {"client.<id>.dataloader": ...}`. Do NOT redesign Rng.load_state_dict — replay-side restoration lands with T3. |
| 4 | MAJOR | Confirmed | Integration test with the REAL `Recorder`: per-client stages load back as lists matching `selected_ids`, arrays bit-identical, and all stage hashes equal across two clean runs. |
| 5 | MAJOR | Confirmed | Spy-registry test asserting exact stream names requested (fail on unexpected names); pre-consume unrelated stream → outputs unchanged; one end-to-end determinism test with the real `falcon.replay.rng.Rng`. |
| 6 | MINOR | Confirmed | `aggregate`: reject duplicate/missing/extra client ids, non-finite or negative weights, total <= 0. Tests for zero-total and NaN weight. |
| 7 | MINOR | Confirmed, contract amended | CONTRACTS now blesses `run(cfg, recorder=None, rng=None)`. Add `"Rng"` (string) annotations to the four stage functions' `rng` params. |
| 8 | MINOR | Confirmed | Hand-computed mean cross-entropy test + extreme-finite-logit case returning finite loss. |

Rules: numpy/pydantic/pyyaml/stdlib only; do not touch falcon/schema, falcon/replay, falcon/recorder; `python -m pytest tests -q` green before finishing; no git commit.
