# T2 adversarial review

Reviewed against `docs/CONTRACTS.md` and `docs/tasks/T2-synthetic-pipeline.md`. The targeted tests pass, but they do not expose the highest-impact failures below.

## Ranked findings

1. **CRITICAL — `falcon/pipeline/runner.py:49-65`: per-client recorder states overwrite one another.**

   **What breaks:** `local` and `compression` are per-client stages, but the runner calls `recorder.record(round_id, stage, state)` once per client. The real `Recorder` treats a single model as the complete stage value and writes it to `round_<n>/local.{json,npz}` or `compression.{json,npz}`. Every later client therefore replaces the previous client. Replays and interventions see only the last selected client, so the recorded run is not a faithful stage-boundary trace.

   **Concrete repro:** With five clients and `clients_per_round=3`, run one round using a real `Recorder`, then load the per-client stages:

   ```python
   rec = Recorder(tmp_path, cfg.run_id)
   run(cfg, recorder=rec)
   print(type(rec.load(0, "local")).__name__)
   print(type(rec.load(0, "compression")).__name__)
   ```

   Both values are a single model (in the reproduced run, only `client_4`), not lists of three models. This is directly incompatible with the recorder's supported per-client list representation.

   **Suggested fix:** Finish each per-client loop first, then call `_record(recorder, round_id, "local", local_states)` and `_record(..., "compression", compressed)` exactly once per stage. Add an integration test using the real `Recorder` that loads each stage and asserts the client lists equal `selection.selected_ids`.

2. **MAJOR — `falcon/pipeline/synthetic_data.py:55-73`: the minority partition implements the opposite conditional distribution from the documented concentration.**

   **What breaks:** `_MINORITY_CONCENTRATION` is documented as the fraction of minority-class samples placed on minority clients, but the code instead makes 90% of each selected client's samples the target class while leaving every other client uniform. For two classes and a 20% client subset, the target class becomes globally *more common*, and most target-class examples remain outside the designated subset. This invalidates experiments intended to diagnose minority-client/class behavior.

   **Concrete repro:** For 10 clients, 10,000 samples/client, two classes, `minority_class=1`, and `minority_client_fraction=0.2`, seed 123 produced a global class-1 rate of `0.58989`; the two most class-1-heavy clients contained only `0.32226` of all class-1 samples, not approximately 0.9. Also, `round(num_clients * fraction)` can silently select zero minority clients (for example, two clients at fraction 0.2).

   **Suggested fix:** Define the intended global minority prevalence, generate the class counts first, and allocate the requested share of minority examples to a guaranteed nonempty designated subset while respecting per-client capacities. At minimum, suppress the minority label on non-designated clients rather than leaving them uniform. Add invariant tests for global rarity, concentration share, subset size, determinism from `DatasetConfig.seed`, and float64 features.

3. **MAJOR — `falcon/pipeline/stages.py:74-83,99-117`: RNG positions are discarded at the only states that have `rng_state`.**

   **What breaks:** `SelectionState` and `ClientLocalState` are returned with their default `rng_state={}` even after consuming `client_selection` and `client.<id>.dataloader`. The contract provides `Rng.state_dict()` specifically for recording/replay, and these schema fields are the boundary representation for it. A recorded boundary therefore cannot restore the random position needed to continue or intervene deterministically; its content hash also does not commit to the RNG position that determines later rounds.

   **Concrete failure:** A direct call to either stage with the production `Rng` yields `{}`. Passing that value to `Rng.load_state_dict()` raises `ValueError`, so replay cannot resume from the recorded state.

   **Suggested fix:** Snapshot `rng.state_dict()` after each consuming stage and store it in the returned state. Extend the inline test stub with `state_dict`, then verify restoration produces the exact next draw. If the intended contract is instead to store only a single stream state, document that representation and provide a matching restoration API; an unexplained empty dict is not replay state.

4. **MAJOR — `tests/integration/test_runner.py:66-84`: the fake-recorder test asserts the broken protocol and masks lost clients.**

   **What breaks:** The fake accepts arbitrary repeated `(round_id, stage, state)` calls and the assertions require three separate `local` and `compression` calls. The production recorder expects one list for a per-client stage. Thus this test passes precisely when the real recorder loses two of three selected clients. The integration suite also never compares real recorded content hashes at all stage boundaries, despite the determinism contract.

   **Concrete failure:** The complete targeted suite reports 12 passes while the repro in finding 1 loads only one of three local/compression states.

   **Suggested fix:** Replace or supplement the fake with a real `Recorder(tmp_path, run_id)`. Assert one complete logical boundary per stage, list length and client IDs for per-client stages, bit-identical arrays after load, and equality of every stage hash across two clean runs.

5. **MAJOR — `tests/unit/test_stages.py:58-83` and `tests/integration/test_runner.py:18-29,58-63`: determinism tests do not enforce the named RNG registry contract.**

   **What breaks:** The tests compare two calls using identical permissive stubs, but never assert which stream name was requested, independence from unrelated streams, or behavior with the production `Rng`. `select_clients` could request `"selection"` instead of `"client_selection"`; `local_train` could request the optimizer stream; or either function could instantiate a fixed local generator. The current assertions would still pass because they check repeatability, not registry compliance.

   **Concrete mutant that survives:** Change `rng.stream("client_selection")` to `rng.stream("wrong-name")`. Selection remains deterministic, uniform, and without replacement, so `test_select_clients_deterministic_and_uniform` still passes. The same weakness applies to the local stream name.

   **Suggested fix:** Use a spy registry that records exact requested names and fails on unexpected names. Pre-consume unrelated streams and prove outputs are unchanged, pre-consume the relevant stream and prove they change, and run at least one end-to-end determinism test with `falcon.replay.rng.Rng` rather than a different seed-derivation implementation.

6. **MINOR — `falcon/pipeline/stages.py:146-159`: `weighted_mean` silently emits NaNs for invalid weights.**

   **What breaks:** There is no check that every received client has one finite, nonnegative weight or that the total is positive. Two zero weights execute `raw / raw.sum()` and return `aggregate=[nan, ...]` plus NaN state weights. Negative, NaN, infinite, and duplicate-client cases can likewise create nonsensical aggregation state instead of failing at the boundary.

   **Concrete repro:** `aggregate([state_a, state_b], {"a": 0.0, "b": 0.0}, AggregationConfig(rule="weighted_mean"), rng)` returns NaNs under NumPy's ordinary warning behavior.

   **Suggested fix:** Reject duplicate client IDs, missing/extra weights, non-finite or negative weights, and totals `<= 0` before normalization. Add zero-total and non-finite tests alongside the existing correct hand-computed happy path.

7. **MINOR — `falcon/pipeline/stages.py:70-72,86-93,120,135-140` and `falcon/pipeline/runner.py:24`: public signatures do not match the contract exactly.**

   **What breaks:** All four stochastic stage functions omit the required `rng: Rng` annotation, and `run` exposes an extra `rng=None` parameter even though the specified API is `run(cfg: RunConfig, recorder=None)`. Runtime callers using the contracted arguments still work, but introspection, generated documentation, static checking, and interface conformance tests see different APIs.

   **Concrete repro:** `inspect.signature(run)` reports `(cfg: RunConfig, recorder=None, rng=None)`; `inspect.signature(select_clients)` reports an unannotated `rng`.

   **Suggested fix:** Annotate against `Rng` (a deferred annotation can avoid import-cycle concerns). Keep RNG injection in a private helper or construct the production `Rng` in the public two-argument `run` entry point.

8. **MINOR — `tests/unit/test_stages.py:158-171`: the evaluation test does not test log-loss correctness or numerical stability.**

   **What breaks:** The only loss assertion is `loss > 0.0`. An implementation returning a constant positive number, using the wrong class probability, or summing instead of averaging would pass as long as prediction accuracy remains correct. No extreme-logit case verifies the intended stable log-softmax behavior.

   **Concrete mutant that survives:** Replace the computed evaluation loss with `1.0`; every assertion in `test_evaluate_on_separable_data` still passes.

   **Suggested fix:** Add a tiny hand-computed logits case with a known mean cross-entropy and an extreme finite-logit case that must return a finite loss. Assert exact per-class denominators, including a deliberately absent class if the chosen zero convention is intended.

## What held up

The normal weighted-mean calculation is correct and has a meaningful hand-computed test. Local training returns a true delta and does not mutate the global parameters; identity compression copies float64 updates exactly. The softmax implementation uses max subtraction, generated feature/model arrays are float64, evaluation data derives only from the dataset seed, selected clients are canonically sorted, and no pipeline code imports Torch or uses wall-clock/OS randomness. The reference YAML loads and the example runs successfully.

Verification: the targeted stage/runner tests passed (`12 passed`); the full suite passed with a workspace-local pytest temp root (`20 passed`); `experiments/run_synthetic.py` completed all ten rounds. All scratch artifacts were removed.
