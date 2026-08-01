# Task T3 — Paired Run Matcher

**Assignee:** Codex
**Contract:** docs/CONTRACTS.md; Plan.md §11.3, §12. You own `falcon/matcher/` and may read (not modify) `falcon/recorder/`, `falcon/replay/`, `falcon/schema/`, `falcon/pipeline/`.

## Deliverables

### 1. `falcon/matcher/matcher.py`

`validate_pair(reference_dir: Path, failure_dir: Path) -> PairValidationReport`

Both dirs are `Recorder` run directories. Checks (each a named bool in `report.checks`):

- `metadata_loadable` — both `metadata.json` parse into `RunMetadata`.
- `same_seed`, `same_rounds`, `same_dataset_config` — exact equality.
- `config_delta_is_failure_only` — the two `RunMetadata.config` dicts differ ONLY in the `failure` entry; reference must have `failure=None`, failure run must have a `FailureSpecification`. Any other config difference → INVALID_PAIR.
- `stage_hash_coverage` — both runs recorded the same set of (round, stage) boundaries.
- `pre_failure_hashes_match` — for every boundary strictly BEFORE the failure's `active_rounds[0]`, stage hashes must be identical. Any pre-failure mismatch → INVALID_PAIR (uncontrolled nondeterminism).
- `selection_matches_pre_failure` — selected_ids identical before the failure window.

Also fill `first_divergence_round` / `first_divergence_stage`: the earliest boundary (round-major, stage order = STAGES) where hashes differ. None if fully identical (then add warning "runs are identical — no failure effect recorded").

Status logic: any hard check fails → `INVALID_PAIR`; all pass, warnings present → `MATCHED_WITH_WARNINGS`; else `MATCHED`.

Warnings (non-fatal): identical runs; divergence starting EARLIER than `active_rounds[0]` at the failure's own stage in the same round is expected — but divergence at a DIFFERENT stage in the first divergent round gets a warning naming both.

### 2. CLI: `falcon/matcher/__main__.py`

`python -m falcon.matcher --reference runs/ref_001 --failure runs/fail_001 [--json out.json]`
Prints status + checks table + first divergence to stdout; `--json` dumps the report via pydantic.

### 3. Tests: `tests/unit/test_matcher.py`, `tests/integration/test_matcher_pipeline.py`

- Unit: synthetic run dirs built with the real `Recorder` — matched pair → MATCHED; corrupted seed → INVALID_PAIR; extra config difference → INVALID_PAIR; pre-failure hash mismatch → INVALID_PAIR; identical runs → warning path.
- Integration: two real `run()` executions with the same `RunConfig` except `failure` field present-but-inert (e.g. active_rounds beyond total rounds) → MATCHED + identical-runs warning; and a manual mid-run tamper (rewrite one recorded array in one round) → correct first_divergence.

## Rules

numpy/pydantic/stdlib only. Do not modify other packages' code or tests. `python -m pytest tests -q` green. No git commit.
