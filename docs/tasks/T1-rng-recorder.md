# Task T1 — RNG registry + Recorder + clean-replay test

**Assignee:** Codex
**Contract:** docs/CONTRACTS.md §3–§5 (read it first, follow names exactly)
**Base:** `falcon/schema/` already exists (pydantic states/configs). Do NOT modify schema files; if something is missing, note it in the PR summary instead.

## Deliverables

### 1. `falcon/replay/rng.py` — class `Rng`

- `Rng(root_seed: int)`; streams created lazily by name via `stream(name: str) -> np.random.Generator`.
- Stream seeding must be **order-independent**: derive per-stream seed from `(root_seed, name)` deterministically (e.g. SeedSequence spawn_key from sha256 of the name). Requesting streams in a different order must yield identical sequences.
- `state_dict() -> dict[str, Any]` / `load_state_dict(d)` — round-trips generator states exactly (use `Generator.bit_generator.state`).
- Standard stream names documented in CONTRACTS §3.

### 2. `falcon/recorder/recorder.py` — class `Recorder`

- `Recorder(root_dir: Path, run_id: str)`; creates `runs/<run_id>/`.
- `save_metadata(meta: RunMetadata)` → `metadata.json`.
- `record(round_id: int, stage: str, state: BaseModel)`:
  - numpy array fields are extracted to `round_<t>/<stage>.npz`, non-array fields to `round_<t>/<stage>.json` (works for any of the state models — detect `np.ndarray` fields generically, including client-keyed collections: for per-client stages ("local", "compression") accept `list[BaseModel]` and store as `<stage>/<client_id>.json/.npz`).
  - computes `content_hash` = sha256 over canonical bytes (sorted-key JSON + raw array bytes in fixed field order) and writes it into the JSON.
- `load(round_id: int, stage: str) -> BaseModel | list[BaseModel]` — reconstructs the exact pydantic object(s); arrays must be bit-identical.
- `stage_hashes() -> dict[(round_id, stage) or str, str]` for whole-run comparison.
- Helper `falcon/recorder/hashing.py`: `hash_array(a: np.ndarray) -> str`, `hash_model(m: BaseModel) -> str`.

### 3. Tests

- `tests/unit/test_rng.py`: same seed → same streams; different names → different streams; order independence; state_dict round-trip mid-sequence.
- `tests/unit/test_recorder.py`: record/load round-trip bit-identical for each state type; content_hash stable across two saves.
- `tests/replay/test_clean_replay.py`: placeholder that records two synthetic states from the same seed and asserts equal hashes (full pipeline replay test lands with T2 — keep this minimal, marked so it still passes standalone).

## Rules

- Python 3.10+, numpy + pydantic + stdlib only. No torch, no new dependencies.
- float64 arrays; no wall-clock, no `random`/`os.urandom` anywhere.
- `pytest` must pass from repo root.
