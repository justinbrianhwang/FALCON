# Task T4 — Failure injectors (one per stage) + runner hook

**Assignee:** Kimi
**Contract:** docs/CONTRACTS.md; Plan.md §10, §35 Task 4. You own `falcon/failures/` and `falcon/pipeline/`; read-only elsewhere.

## Design (PM-decided)

`falcon/failures/base.py`:

```python
class FailureInjector:
    """Built from a FailureSpecification; every transform is identity when inactive."""
    def __init__(self, spec: FailureSpecification, partition: dict[str, ClientData], rng: "Rng"): ...
    def active(self, round_id: int) -> bool          # active_rounds inclusive
    # stage-specific transforms, called by the runner at the marked hook site:
    def candidate_pool(self, pool: list[str], round_id: int) -> list[str]
    def local_cfg(self, client_id: str, cfg: LocalConfig, round_id: int) -> LocalConfig
    def compression_cfg(self, client_id: str, cfg: CompressionConfig, round_id: int) -> CompressionConfig
    def weights(self, weights: dict[str, float], round_id: int) -> dict[str, float]

def build_injector(spec, partition, rng) -> FailureInjector   # dispatch by spec.stage/spec.type
```

All injector randomness comes from stream `failure.<stage>` — NEVER a shared stream (Plan §12.2: injectors must not desynchronize downstream streams). Transforms return NEW objects; never mutate inputs.

## The four failures (Plan §35 suggested set)

1. **selection / `minority_exclusion`** (S1) — params: `target_class: int`, `exclusion_probability: float`. A client is "minority-heavy" if its share of `target_class` samples exceeds the uniform share (compute from `partition` once, deterministically). While active, each minority-heavy client is dropped from the candidate pool independently with `exclusion_probability`, drawn from `failure.selection`.
2. **local / `lr_misconfig`** (L1) — params: `affected_clients: list[str] | fraction: float`, `lr_multiplier: float` (e.g. 10.0, 0.01; negative allowed as sanity case). Affected set chosen once (deterministically from `failure.local` if `fraction` given). Returns cfg copy with `lr * multiplier`.
3. **compression / `aggressive_topk`** (C1) — params: `k_ratio: float` (keep top k fraction by |value|), `affected_clients` optional (default all). Requires implementing the `topk` kind in `stages.compress`: keep top `ceil(k_ratio * n)` coordinates by magnitude, zero the rest, exact float64, deterministic tie-break (larger index wins ties — document it). `bytes_transmitted = nnz * 8 + nnz * 4`. Injector swaps cfg to `topk` with the aggressive ratio while active.
4. **aggregation / `wrong_sample_weights`** (A1) — params: `mode: "uniform" | "swapped" | "corrupted"`. uniform: all weights equal; swapped: reverse the weight values across sorted client ids; corrupted: multiply each weight by a factor drawn from `failure.aggregation` (log-uniform in [0.1, 10]). Weights are re-normalized by `aggregate` as usual.

## Runner wiring

At the marked hook site in `runner.py`: if `cfg.failure` is not None, `build_injector(...)` once per run, apply the four transforms at their stage boundaries. `RunMetadata.failure` already records ground truth — no extra logging. Reference runs (`failure=None`) must execute byte-identically to today (regression: stage hashes unchanged vs current main for the reference config).

## Tests

- `tests/unit/test_failures.py` — per injector: inactive rounds are identity (object-equal output); active behavior matches params (exclusion actually removes minority-heavy clients at p=1.0; lr scaled; topk keeps exactly ceil(k_ratio*n) nonzeros incl. tie-break; each weights mode); determinism (same spec+seed twice → identical transforms); `failure.<stage>` is the ONLY stream consumed (spy Rng).
- `tests/integration/test_failure_runs.py` — for each of the four failures: reference + failure run with same seed → pre-failure-window stage hashes identical, first divergent stage == the injected stage, and the failure run's primary metric is measurably worse (choose severities that visibly degrade; document chosen params in the test).
- Update `configs/cases/`: one YAML per failure (`synthetic_selection_failure.yaml`, etc.) mirroring `synthetic_reference.yaml`.

## Rules

numpy/pydantic/pyyaml/stdlib only. Do not touch falcon/schema, falcon/replay, falcon/recorder, falcon/matcher (Codex is building matcher concurrently — do not create or edit `tests/*matcher*`). `python -m pytest tests -q` green except matcher tests if they appear mid-flight. No git commit.
