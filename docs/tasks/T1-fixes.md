# Task T1-F — Fix findings from adversarial review

**Assignee:** Codex (original T1 author)
**Input:** docs/reviews/T1-review-by-kimi.md — PM verdicts + design decisions below.

| # | Severity | PM verdict | Action |
|---|----------|-----------|--------|
| C1 | CRITICAL | Confirmed. **Design decision: sentinel encoding.** | Canonicalize non-finite floats in BOTH `_canonical_json` and `_write_json`: emit `{"__falcon_float__": "NaN" | "Infinity" | "-Infinity"}`; restore to real floats on load. NaN != NaN is fine — hashing compares encodings. Test: OutcomeState with nan/inf metrics records, loads, hash-stable. |
| M1 | MAJOR | Confirmed | `hash_array`: include dtype + shape header: `sha256(a.dtype.str + str(a.shape) + tobytes)`. Regression test with the two collision pairs from the review. |
| M2 | MAJOR | Confirmed. **Design decision: fail loud.** | Reject non-JSON-native values in `Any`-typed fields at record time (tuple, Enum, non-str dict keys) with a clear ValueError naming the field. Test with tuple in `compression_params`. (Note: numpy `bit_generator.state` dicts are JSON-native and must keep working — T2 now stores them in `rng_state`.) |
| M3 | MAJOR | Confirmed | Duplicate client check via `casefold()`; also reject client_id with trailing dots/spaces. Windows-safe by construction, same behavior on all platforms. |
| M4 | MAJOR | Confirmed | Hash-sensitivity tests: one-ulp array change, one scalar change, one dict entry change → different `content_hash`; different seed → different `stage_hashes`; tampered `.json` byte → `load` raises. Also add a direct `hash_array` test (currently untested). |
| m1 | MINOR | Confirmed | Treat a dict as array-reference only if `__falcon_array__` marker AND dtype/shape/path all present; reject the marker key inside user data at record time. |
| m2 | MINOR | Confirmed | `record`/`load`: `TypeError` unless `isinstance(round_id, int)` and not bool. |
| m3 | MINOR | Confirmed, docs-only | Document in `load_state_dict` docstring: previously handed-out generators are stale; re-fetch streams after load. |
| m4 | MINOR | Confirmed | Reject bare (non-list) states for `"local"`/`"compression"` — contract §1 now mandates list recording. Test included. |
| m5 | MINOR | PM fixed CONTRACTS §4 already | Add a `# placeholder — superseded by full pipeline replay test (T2-F)` pytest mark/comment in test_clean_replay.py. |

Also cover the "weak tests" list at the end of the review (snapshot independence of `state_dict`, unknown stage, traversal client_id, list-for-scalar-stage rejection).

Rules: numpy/pydantic/stdlib only; do not touch falcon/schema or falcon/pipeline; `python -m pytest tests -q` green before finishing; no git commit.
