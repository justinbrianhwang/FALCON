# T4/T5/T8 adversarial review — round 2

Reviewed `falcon/failures/`, `falcon/intervention/`, the overlay hook in `falcon/pipeline/runner.py`, `falcon/baselines/`, and the requested tests. The current suite is green, but two controls that underpin the causal claim are not valid under adversarial replay.

## Ranked findings

1. **CRITICAL — `falcon/pipeline/runner.py:63-90`: a selection overlay changes persistent client RNG positions, contaminating later-round effects.**

   **Failure scenario:** The runner first replaces `selection`, then trains the replacement clients using persistent `client.<id>.dataloader` generators. Selecting a client one extra time advances that client's stream; omitting another leaves its stream behind. If either client participates later, its minibatches differ from the matched run even when the model, later selection, seed, and all other exogenous inputs have been restored. Thus a selection restore/inject measures both the state intervention and an implementation-induced change in future minibatch randomness. Named streams prevent cross-client/order coupling, but they do not prevent participation-count coupling.

   **Repro:** With the configuration from `tests/unit/test_overlay.py`, the baseline selects `[client_0, client_2, client_4]` in round 0 and `[client_1, client_2, client_4]` in round 1. A custom overlay replaces round-0 selection with `[client_1, client_2, client_4]` and also replaces the round-0 aggregate with the baseline aggregate, deliberately restoring the model exactly. The observed output was:

   ```text
   round0_models_equal True
   round1_selection_equal True
   client_1 same_update False
   client_2 same_update True
   client_4 same_update True
   ```

   Only `client_1`, whose stream was advanced by the selection intervention, changes in round 1. This isolates RNG consumption from model causation.

   **Suggested fix:** Key stochastic local-training schedules by `(round_id, client_id)` (for example `client.<id>.round.<t>.dataloader`) or pre-record/replay the per-round minibatch indices, so whether a client participated earlier cannot change its later exogenous draws. Add the aggregate-restoring counterexample above as a regression test. Until fixed, do not use multi-round selection interventions as causal evidence.

2. **CRITICAL — `falcon/intervention/engine.py:233-239,249-255`: sham can erase replay drift and falsely report exactly zero artifact.**

   **Failure scenario:** Sham does not test an unmodified replay. It loads the target's recorded boundary and overlays it onto the live replay. At aggregation, that replacement discards every upstream replay difference in the intervention round; at evaluation, it directly replaces the outcome being compared. Consequently `sham_deviation_* == 0` can be true because the sham repaired the drift, not because replay was faithful. The all-stage test in `tests/interventions/test_engine.py:160-173` only exercises an already reproducible target, so it cannot distinguish these cases.

   **Repro:** I recorded a one-round run at `local.lr=0.05`, changed only recorded metadata to `local.lr=50.0`, and compared a plain replay with an aggregation sham:

   ```text
   original loss                0.5898301871803981
   plain replay after drift     1.3246494823635377
   sham valid                   True
   sham_deviation_loss          0.0
   ```

   The replay is grossly different, yet the sham gate certifies zero deviation.

   **Suggested fix:** First run a no-overlay replay and require all recorded boundary hashes (not only final metrics) to match the target. A serialization sham should round-trip the *live* boundary and compare against a pass-through replay; it must not insert the recorded target boundary. Treat evaluation-stage self-replacement as tautological, not replay evidence. Add a deliberately drifted-code/config fixture that the sham gate must reject.

3. **MAJOR — `falcon/intervention/engine.py:131-138,238-248`: incompatible lineage remains `valid=True` and produces restore/inject asymmetry from delta transplantation.**

   **Failure scenario:** A local/compression state produced from a different live lineage is transplanted anyway; the only consequence is a numeric warning inside `outcome_metrics`. The analyzer excludes only `valid=False` interventions, so warned interventions remain scientific evidence. A delta trained/compressed under the reference model and added to an already-diverged failure model is not a restoration of the reference stage state. Reversing direction similarly applies a failure delta to the reference base, so restore and inject need not be symmetric for implementation reasons.

   **Repro:** Using the existing intervention fixture at round 3, where the test itself confirms a lineage mismatch:

   ```text
   recorded ref loss            0.08181304366713268
   recorded fail loss           0.09285147665030627
   restore(valid=True) loss     0.09660575564653728
   inject(valid=True) loss      0.07853288018316090
   warning_base_model_mismatch  1.0 in both
   ```

   “Restore” makes the failed run worse while “inject” improves the reference, precisely the misleading asymmetry the bidirectional score is supposed to guard against. `tests/interventions/test_engine.py:233-241` currently institutionalizes this as valid.

   **Suggested fix:** Return `valid=False` for lineage mismatch, or introduce an explicit `raw_delta_transplant` mode that is excluded from restore/inject attribution and reported separately. If later-round restoration is required, restore a compatible model/checkpoint and its optimizer/RNG state before applying the source boundary.

4. **MAJOR — `falcon/intervention/engine.py:82-90,145-151,215-239`: shape equality is treated as compatibility, so unrelated model layouts are accepted without warning.**

   **Failure scenario:** The engine never validates source metadata against target metadata. Aggregation checks only `np.shape(aggregate)`. Different `(num_classes, num_features)` layouts can have the same flattened parameter length, so arrays with incompatible coordinate meaning pass. Selection/evaluation have no compatibility or lineage check at all. The CLI can therefore label an unmatched cross-run transplant `valid=True`; only the separate reporting driver happens to validate pairs first.

   **Repro:** A target with `(D=5, K=2)` and a source with `(D=3, K=3)` both have `K*(D+1)=12` parameters. One-round recorded aggregation states both had shape `(12,)`. Applying the source aggregation to the target returned:

   ```text
   aggregate_shapes (12,) (12,)
   accepted True
   warning None
   ```

   **Suggested fix:** Load source metadata and require the same seed/dataset/model layout and the permitted config delta before replay, ideally by requiring a `MATCHED`/allowed-warning pair report. Add a typed model-layout fingerprint to every model-bearing boundary and validate it in addition to shape, dtype, finiteness, round, and client identities. Parameterize shape/compatibility tests over local, compression, and aggregation; the current shape test (`tests/interventions/test_engine.py:206-210`) covers compression only.

5. **MAJOR — `falcon/intervention/engine.py:170-177,201-241`: documented validation paths still raise ordinary exceptions.**

   **Failure scenario:** `apply_intervention` catches only `_InvalidIntervention`. `scope["client_ids"]` is untyped `Any`; `None` raises `TypeError` at line 213. `Recorder.load` integrity/type errors other than `FileNotFoundError` escape `_load_recorded`. Invalid stage-state model types can cause `AttributeError`, an out-of-range recorded round can later cause `IndexError`, and unsafe run IDs can cause `ValueError`. These contradict the function's “never raises for validation failures” contract. An empty client list is also accepted as a valid no-op intervention, which can silently enter attribution results.

   **Repro:** Against the existing valid pair:

   ```python
   spec = InterventionSpecification(
       target_run_id="fail", source_run_id="ref", round_id=2,
       stage="local", mode="restore", scope={"client_ids": None},
   )
   apply_intervention(spec, root)
   # TypeError: 'NoneType' object is not iterable
   ```

   `scope={"client_ids": []}` instead returns `valid=True` with the unchanged failure loss.

   **Suggested fix:** Validate scope as a nonempty list of unique nonempty strings; validate `0 <= round_id < cfg.rounds` before loading; translate recorder/Pydantic/path/type failures into stable invalid reasons; and verify the overlay actually fired exactly once before indexing outcomes. Add corrupt-hash, wrong-state-type, `None`/string/empty/duplicate scope, and rogue recorded-round tests.

6. **MAJOR — `falcon/failures/local/lr_misconfig.py:21-24` and `falcon/baselines/passive.py:38-49,127-147`: non-finite failures silently force the passive rival to the first stage.**

   **Failure scenario:** `lr_multiplier` accepts `NaN`/infinity. This creates non-finite local, compression, and aggregation states. `_relative_l2` returns `NaN`, `_mean` preserves it, and Python's `max` never promotes a later `NaN` key over the first finite key. The ground-truth local failure is therefore localized as `selection`. This is not merely bad input handling: it silently weakens the passive baseline used to support the paper comparison. `NearestCentroidStageClassifier.predict` has the same first-class bias because `np.argmin` over NaN distances returns index 0.

   **Repro:** A one-round local failure affecting every client with `lr_multiplier=float("nan")` produced:

   ```text
   {'selection': 0.0, 'local': nan, 'compression': nan, 'aggregation': nan}
   passive_localize(...) == 'selection'
   ```

   **Suggested fix:** Reject non-finite injector parameters and non-finite baseline features/scores with a declared invalid-sample result; never pass them to `max`/`argmin`. If non-finite state is an intentional failure class, define a finite maximum anomaly score for it consistently across stages. Add NaN/inf tests for `_relative_l2`, `passive_localize`, classifier fit/predict, and `lr_multiplier`.

7. **MAJOR — `falcon/failures/selection/minority_exclusion.py:59-68` with `falcon/pipeline/runner.py:63-66`: a valid exclusion can shrink the pool below the requested sample size and crash the run.**

   **Failure scenario:** The injector has no knowledge of `clients_per_round`, and the runner passes the reduced pool directly to NumPy sampling without a feasibility check. A strong but otherwise valid selection failure becomes an infrastructure exception rather than a recorded failed execution, biasing the benchmark toward only mild/executable selection failures.

   **Repro:** Ten clients, `minority_client_fraction=0.8`, `clients_per_round=5`, `exclusion_probability=1.0`, and an active round produced:

   ```text
   ValueError: Cannot take a larger sample than population when replace is False
   ```

   **Suggested fix:** Define the failure semantics explicitly: either retain enough clients deterministically, allow undersubscribed selection and record it as such, or reject the failure/config combination before the run with a domain-specific validation error. Add an integration test where excluded clients exceed `num_clients - clients_per_round`.

8. **MAJOR — `tests/unit/test_overlay.py:64-72` and `tests/interventions/test_engine.py:160-241`: the replay-validity tests have no independent oracle and several stage mutants survive.**

   **Failure scenario / surviving mutants:** `test_overlay_none_is_byte_identical` runs the current runner twice and compares it to itself; it does not compare against pre-overlay golden hashes. An unconditional extra RNG draw or common runner behavior change affects both sides and passes. Sham tests use only an already deterministic pair, so the false-zero implementation in finding 2 passes. Removing aggregation shape validation survives because the sole engine shape test uses compression. Removing local-lineage validation survives because the only mismatched-lineage assertion uses compression. The baseline tests cover zero norm but not non-finite values, so finding 6 also survives the full suite.

   **Suggested fix:** Commit a small golden boundary-hash fixture from the pre-hook runner; add the RNG-isolating selection-overlay test from finding 1 and the drifted sham test from finding 2; parameterize engine validation across every applicable stage and both directions; and add non-finite baseline cases. For failure-strength tests, pin expected RNG outputs/distribution parameters rather than only checking range and repeatability (`tests/unit/test_failures.py:434-456`).

## What held up

- Failure injectors request only `failure.<stage>` streams; the registry derives streams independently by name, so constructing or drawing an injector stream does not advance selection, client, aggregation, or evaluation streams.
- Inactive transforms return equal new objects and do not mutate inputs. Deterministic top-k validates finite ratios, has a defined tie-break, and does not consume RNG.
- Corrupted aggregation weights iterate clients in sorted-ID order and flow through the normal finite/nonnegative renormalization checks.
- On compatible states at the first divergent compression boundary, restore and inject use the same replacement machinery and reproduce the opposite run exactly.
- Whole-stage list replacement enforces ordered client identity; scoped replacement matches by exact ID and correctly counts missing clients as invalid. Passive unmatched-client scoring follows the stated maximum-penalty rule.
- Terminal features have documented stable ordering for normal canonical class keys, training-only z-scaling, and a zero-variance guard. No evidence of ordinary finite-feature scaling leakage was found.

## Verification

- Requested target suite: `79 passed in 3.37s` using a workspace-local pytest temp root.
- Full suite: `187 passed in 19.88s`.
- Repros were run as inline throwaway scripts; system-temp run artifacts were removed. No source or test file was modified and no commit was made.
