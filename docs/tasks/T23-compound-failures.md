# T23 — Compound (multi-stage) failures + origin_set experiment

Owner: Codex.

## Goal

Plan §10.5 secondary benchmark and §15.3 ambiguity handling: inject failures at TWO
stages simultaneously and verify FALCON returns an honest answer (`origin_set` /
`unresolved` / first-origin with carriers) instead of fabricating a unique origin.
The machinery (AttributionReport.outcome, origin_set) exists but has never been
exercised by a real compound run.

## Design (PM-decided; keep exactly this shape)

1. `falcon/schema/config.py` — `RunConfig` gains
   `failures: list[FailureSpecification] = []` (default empty, pydantic Field).
   The existing single `failure: Optional[FailureSpecification]` stays and keeps
   its exact semantics. Reject configs setting BOTH (pydantic model_validator,
   fail loud). `RunMetadata` (falcon/schema/states.py) gains the same optional
   `failures` list so recorded metadata round-trips.
2. `falcon/pipeline/runner.py` — build one injector per failure spec
   (`build_injector` per spec, same partition/rng) and CHAIN them in list order:
   candidate_pool, local_cfg, compression_cfg, weights each pass through every
   injector in sequence. Single-failure behavior must stay byte-identical
   (reference and existing failure runs must reproduce their stage hashes — run
   `pytest tests/replay tests/integration/test_e0_smoke.py` to prove it).
   Failures at DIFFERENT stages draw from different `failure.<stage>` RNG streams
   (CONTRACTS section 3), so chaining introduces no stream coupling; same-stage
   compound is OUT OF SCOPE — raise ValueError if two specs share a stage.
3. `falcon/matcher/matcher.py` — the `config_delta_is_failure_only` check must
   accept a delta confined to `failure` OR `failures`. Read the current
   implementation first and extend minimally.
4. `falcon/reporting/analyze.py::load_ground_truth` etc. must not crash on
   compound metadata (failure=None, failures=[...]). analyze_pair's
   window derivation uses `failure.active_rounds`; for compound runs use the
   UNION window (min start, max end) over `failures`. Keep the single-failure
   code path untouched.
5. `experiments/run_compound.py` (pattern: experiments/run_coauthor_cifar.py):
   synthetic Tier-0, reference from `configs/cases/synthetic_reference.yaml`,
   two compound cases with specs copied from the calibrated
   `configs/cases/synthetic_*_failure.yaml` files:
     - selection+compression (S1 sev 2 + C1 sev 2)
     - selection+aggregation (S1 sev 2 + A1 biased variant from
       `configs/cases/synthetic_aggregation_biased.yaml`)
   `--smoke` = 4 rounds. analyze_pair on accuracy, min_gap=0.005,
   sham_tolerance=1e-9. Summary row per case: outcome, origin_ranking,
   origin_set, notes. Do NOT tune anything to force unique origins — an honest
   `origin_set`/`unresolved` IS the expected headline result.

## Tests (required)

- Unit: compound RunConfig validation (both-set rejection, same-stage rejection).
- Integration: a 4-round synthetic compound run records hashes that differ from
  reference at BOTH injected stages within the active window; a single-failure
  run under the new code reproduces the exact stage hashes it records today
  (regression guard — compare against a run made with failure= single spec).
- Full `pytest` green.

## Acceptance

- `python experiments/run_compound.py --smoke` exits 0, writes
  `results/compound/summary.json` + one report per case.
- Full suite green; no behavior change for single-failure/reference runs.
- cp949-safe stdout (ASCII only in prints).
