# FALCON Stage & Schema Contracts (v0.1)

Single source of truth for names and signatures. **Codex and Kimi both code against this file.**
Changing a name/signature here requires PM sign-off; do not improvise different names.

## 1. Pipeline stages

One round = five pure-as-possible stage functions. All mutable state passes explicitly; no module-level globals. All randomness comes from the named RNG registry (§3).

```python
# falcon/schema/ — pydantic models (Task T1)
RunMetadata, RoundState,
SelectionState, ClientLocalState, CompressionState, AggregationState, OutcomeState,
FailureSpecification, InterventionSpecification, InterventionResult,
PairValidationReport, AttributionReport
```

```python
# Stage interfaces (implemented by the pipeline, Task T2)

def select_clients(pool: list[str], round_id: int, cfg: SelectionConfig,
                   rng: Rng) -> SelectionState: ...

def local_train(model_params: np.ndarray, client_id: str, data: ClientData,
                round_id: int, cfg: LocalConfig, rng: Rng) -> ClientLocalState: ...
    # ClientLocalState.update = delta (trained - global), NOT the trained params

def compress(local_state: ClientLocalState, cfg: CompressionConfig,
             rng: Rng) -> CompressionState: ...
    # identity compression is the default; must round-trip exactly

def aggregate(compressed: list[CompressionState], weights: dict[str, float],
              cfg: AggregationConfig, rng: Rng) -> AggregationState: ...

def evaluate(model_params: np.ndarray, eval_data: EvalData) -> OutcomeState: ...
```

MVP model params are a flat `np.ndarray` (float64) — logistic regression first. Torch enters at Tier 1 (CIFAR); do not import torch in `falcon/schema` or the synthetic pipeline.

## 2. Stage names (string enum, used everywhere)

```
"selection" | "local" | "compression" | "aggregation" | "evaluation"
```

## 3. RNG registry (Plan §12.2)

`falcon/replay/rng.py` provides `Rng`, a registry of **named, independent** `numpy.random.Generator` streams:

```
global_init
client_selection
client.<id>.dataloader
client.<id>.optimizer
compression.<id>
aggregation
evaluation
```

- Constructed from a single root seed via `numpy.random.SeedSequence.spawn` keyed by stream name (order-independent: hashing the name, not creation order).
- `Rng.state_dict()` / `Rng.load_state_dict()` for recording and replay.
- A failure injector must use its **own** stream (`failure.<stage>`), never a shared one.

## 4. Recorder contract

- `Recorder.record(round_id: int, stage: str, state: BaseModel) -> None`
- Storage: one run directory `runs/<run_id>/` with `metadata.json`, per-round `round_<t>/<stage>.json` (+ `.npz` for arrays).
- Every recorded state carries `content_hash` (sha256 of canonical serialized bytes).
- `Recorder.load(run_id, round_id, stage)` returns the pydantic object back, bit-identical arrays.

## 5. Determinism ground rules

- float64 numpy only in the synthetic pipeline; no threading; no wall-clock or os randomness.
- Two runs with the same config + seed must produce identical `content_hash` at every stage boundary (this is test `tests/replay/test_clean_replay.py`).

## 6. Config objects

Plain pydantic models in `falcon/schema/config.py`: `SelectionConfig`, `LocalConfig`, `CompressionConfig`, `AggregationConfig`, `RunConfig` (composes the four + dataset + seed + rounds). YAML in `configs/` maps 1:1 onto `RunConfig`.
