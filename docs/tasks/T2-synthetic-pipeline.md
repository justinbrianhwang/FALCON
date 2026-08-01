# Task T2 — Deterministic synthetic FL pipeline (skeleton)

**Assignee:** Kimi
**Contract:** docs/CONTRACTS.md §1–§3, §5–§6 (read it first, follow names/signatures exactly)
**Base:** `falcon/schema/` already exists (pydantic states/configs). Do NOT modify schema files; if something is missing, note it in your summary instead.

**Coordination:** Codex is building `falcon/replay/rng.py` (`Rng` class, CONTRACTS §3) and the recorder in parallel. Code against that interface but DO NOT create `falcon/replay/rng.py` or anything in `falcon/recorder/` yourself. For your tests, use a minimal inline stub with the same `stream(name) -> np.random.Generator` method.

## Deliverables

### 1. `falcon/pipeline/synthetic_data.py`

- `make_partition(cfg: DatasetConfig) -> dict[str, ClientData]` — synthetic logistic-regression data (Gaussian class clusters), partitioned across `num_clients`.
- `ClientData`: simple dataclass `(x: np.ndarray, y: np.ndarray)`; also `EvalData` alias.
- `heterogeneity` shifts client feature means; `minority_class`/`minority_client_fraction` concentrates one class on a client subset.
- Partition depends ONLY on `cfg.seed` (own `np.random.Generator`), never on the run seed.

### 2. `falcon/pipeline/stages.py`

Implement the five stage functions exactly as in CONTRACTS §1:

- `select_clients`: uniform sampling without replacement of `clients_per_round`, using stream `client_selection`; fills `SelectionState` (probs = uniform).
- `local_train`: logistic regression, plain minibatch SGD, `local_steps` steps, minibatches drawn from stream `client.<id>.dataloader`; returns `ClientLocalState` with `update = trained - global`, `loss_history` per step, `base_model_hash` (sha256 of param bytes).
- `compress`: `identity` kind only for now — copies the update, `bytes_transmitted = update.nbytes`; structure ready for `topk`/`quantization` later (raise `NotImplementedError` for those kinds).
- `aggregate`: `weighted_mean` by `num_examples` (weights arg) and `uniform_mean`; others `NotImplementedError`. Fills `AggregationState`.
- `evaluate`: accuracy + mean log-loss on `EvalData`, per-class accuracy into `OutcomeState.per_class`.
- Model: flat float64 param vector for multinomial/binary logistic regression (include bias). Pure numpy, no torch.

### 3. `falcon/pipeline/runner.py`

- `run(cfg: RunConfig, recorder=None) -> list[OutcomeState]` — the round loop: select → local (each selected client) → compress → aggregate → apply update (`w += aggregate`) → evaluate. If `recorder` is not None call `recorder.record(round_id, stage, state)` at every boundary (duck-typed; don't import the Recorder class).
- Global eval set generated from `DatasetConfig` with a fixed derived seed.
- No failure injection yet — but leave a clearly marked single call site where a `FailureSpecification` hook will wrap each stage (`falcon/failures/` comes in T4).

### 4. Tests + example

- `tests/unit/test_stages.py`: each stage is deterministic given a seeded generator stub; identity compression round-trips exactly; weighted_mean matches a hand-computed value.
- `tests/integration/test_runner.py`: 5 clients, 3 rounds, loss decreases from round 0 to final; two `run()` calls with the same `RunConfig` produce identical final params.
- `configs/cases/synthetic_reference.yaml` matching `RunConfig`, and `experiments/run_synthetic.py` that loads it and prints per-round accuracy.

## Rules

- Python 3.10+, numpy + pydantic + pyyaml + stdlib only. No torch, no sklearn, no new dependencies.
- float64 only; all randomness through the provided rng streams; no wall-clock.
- `pytest` must pass from repo root (excluding Codex's directories if absent).
