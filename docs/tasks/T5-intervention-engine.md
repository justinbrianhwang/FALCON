# Task T5 — Intervention engine: Restore / Inject / Sham + replay

**Assignee:** Kimi
**Contract:** docs/CONTRACTS.md; Plan.md §11.5, §13.1–13.3, §12.4. You own `falcon/intervention/` and `falcon/pipeline/`; read-only elsewhere.

## PM design decision — replay strategy

MVP replay = **full deterministic re-execution with an overlay**, NOT checkpoint restoration.
Runs are cheap and bit-deterministic; re-run `run(cfg)` from round 0 and, at the intervention
boundary, swap the produced stage state for the recorded source state, then continue downstream.
Checkpointed suffix replay is a later optimization (Plan R4). Do not build it now.
<!-- ponytail: full re-execution replay; checkpointed suffix replay when runs get expensive -->

## Deliverables

### 1. Runner overlay hook (`falcon/pipeline/runner.py`)

`run(cfg, recorder=None, rng=None, overlay=None)` — `overlay` duck-typed:
`overlay.override(round_id: int, stage: str, state)` returns the (possibly replaced) state the
downstream pipeline must consume. Called at every stage boundary AFTER the stage computes and
AFTER failure injection, BEFORE recording and before downstream use. Default None = today's
behavior, byte-identical (regression-checked).

Note the consumption semantics per stage: replacing `selection` changes which clients train;
replacing `local`/`compression` (list states) changes what aggregation sees; replacing
`aggregation` changes the model update; replacing `evaluation` only changes the recorded outcome.

### 2. `falcon/intervention/engine.py`

`apply_intervention(spec: InterventionSpecification, runs_root: Path) -> InterventionResult`

- Load target run's `RunMetadata` from `runs_root/<target_run_id>` → reconstruct its `RunConfig`.
- Load source state = `Recorder(runs_root, spec.source_run_id).load(spec.round_id, spec.stage)`.
- Modes:
  - `restore` / `inject`: identical machinery (direction lives in which run is target/source).
  - `sham`: ignore source_run_id content-wise — take the TARGET's own recorded state at that
    boundary, round-trip it through serialization (`Recorder` save/load in a temp dir), and use
    that as replacement. Outcome must match the unmodified target run; report the deviation.
- Scope: `{}` = whole stage. `{"client_ids": [...]}` valid only for `local`/`compression`:
  replace only those clients' entries in the list, keep the rest from the live replay.
- Validation (fail → `InterventionResult(valid=False, reason=...)`, never raise for these):
  - target/source runs exist and recorded that (round, stage);
  - array shape mismatch vs the live computed state;
  - scoped client_ids not present in BOTH live and source state;
  - `local`/`compression` source entries whose `base_model_hash`/`uncompressed_hash` lineage
    does not match the live replay's — record `"base_model_mismatch"` as a WARNING inside
    `outcome_metrics` (key `warning_base_model_mismatch: 1.0`) but proceed (Plan §13: replacing
    cross-run states is the point; the hash difference is expected and must be surfaced, not fatal).
- Re-execute with overlay; `InterventionResult.outcome_metrics` = final round's
  `OutcomeState.metrics` plus `"round_<t>_<metric>"` for the intervention round.

### 3. CLI: `falcon/intervention/__main__.py`

Mirror Appendix B: `python -m falcon.intervention --runs-root runs --target-run fail_001 --source-run ref_001 --round 1 --stage compression --mode restore [--client-ids a,b] [--json out.json]`.

### 4. Tests

- `tests/unit/test_overlay.py` — overlay=None byte-identical (hash regression); a recording
  overlay sees all 5 boundaries; replacement at each stage type actually propagates downstream.
- `tests/interventions/test_engine.py` — build a reference+failure pair (reuse a T4 failure),
  then: restore at the injected stage recovers the reference-level metric (and restore at a
  pre-divergence bystander stage does NOT); inject the failed state into the reference degrades
  it; sham at every stage produces zero metric deviation; each validation failure path returns
  valid=False with the right reason; scoped client replacement replaces exactly those clients.

## Rules

numpy/pydantic/pyyaml/stdlib only. Do not touch falcon/schema, falcon/replay, falcon/recorder,
falcon/matcher, falcon/attribution, falcon/metrics (Codex works there concurrently).
`python -m pytest tests -q` green (except attribution tests if they appear mid-flight). No git commit.
