# T1 Adversarial Review — RNG registry + Recorder (by Kimi)

Scope: `falcon/replay/rng.py`, `falcon/recorder/recorder.py`, `falcon/recorder/hashing.py`,
`tests/unit/test_rng.py`, `tests/unit/test_recorder.py`, `tests/replay/test_clean_replay.py`,
against `docs/CONTRACTS.md` §3–§5 and `docs/tasks/T1-rng-recorder.md`.

Environment: Python 3.13.11, numpy 2.5.1, pydantic 2.13.4, Windows (NTFS, case-insensitive).
Every claim below was reproduced with a throwaway script (since deleted); snippets inline.
Baseline: all 20 existing tests pass.

---

## CRITICAL

### C1. Recorder refuses NaN/Inf scalar metrics — the signature artifact of failure runs

- **Where:** `falcon/recorder/hashing.py:68-74` (`allow_nan=False` in `_canonical_json`),
  also `falcon/recorder/recorder.py:222` (`allow_nan=False` in `_write_json`).
- **What breaks:** `Recorder.record()` raises `ValueError` for any state containing a NaN/Inf
  **scalar** float, while NaN inside numpy arrays is recorded fine. FALCON exists to record and
  compare *failure* runs; a diverged run's `OutcomeState.metrics = {"loss": nan}` is the single
  most common post-failure state. The pipeline will crash at the evaluation boundary of exactly
  the runs the tool was built for — instead of recording them.
- **Repro:**
  ```python
  st = OutcomeState(round_id=0, model_hash="m", metrics={"loss": float("nan")})  # constructs fine
  Recorder(tmp, "r").record(0, "evaluation", st)
  # ValueError: Out of range float values are not JSON compliant: nan
  # ...but AggregationState(aggregate=np.array([np.nan])) records & round-trips fine.
  ```
- **Suggested fix:** Decide explicitly (PM/design call), then either canonicalize non-finite
  scalars deterministically (e.g. emit `"NaN"`/`"Infinity"` sentinels in both `_canonical_json`
  and `_write_json`, restoring them on load), or document the rejection as intentional and make
  it consistent (also reject NaN arrays). Silent inconsistency between scalar and array NaN is
  the worst option, and it is the current one.

---

## MAJOR

### M1. `hash_array` omits dtype and shape — different arrays collide

- **Where:** `falcon/recorder/hashing.py:16-20`.
- **What breaks:** The digest covers only `a.tobytes(order="C")`. Arrays with different
  dtype or shape but identical raw bytes hash identically. `hash_array` is exported in
  `falcon/recorder/__init__.py` and is the obvious tool for the schema's provenance fields
  (`base_model_hash`, `uncompressed_hash`, `model_hash`) — i.e. it will be used to assert
  "this update is the one I sent". That assertion is false for cross-dtype/shape pairs.
  (Note `hash_model` itself is safe: `_split_value` records dtype+shape in the JSON.)
- **Repro (all print `True`):**
  ```python
  hash_array(np.zeros(1, np.float64)) == hash_array(np.zeros(8, np.uint8))
  hash_array(np.arange(4, dtype=np.int64)) == hash_array(np.arange(4, dtype=np.int64).reshape(2, 2))
  ```
- **Suggested fix:** Hash a canonical header plus bytes, e.g.
  `sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes(order="C"))`
  (mirroring the metadata `_split_value` already writes). Add a regression test.

### M2. `dict[str, Any]` fields are silently mutated on round-trip; `content_hash` hides it

- **Where:** `falcon/recorder/hashing.py:46-50` (tuple→list), plus JSON normalization of
  nested non-str dict keys and `Enum→value` at `hashing.py:41-54`.
- **What breaks:** Contract §4 promises `load` "returns the pydantic object back". For
  `rng_state` / `compression_params` / `config`-style `Any` fields it does not: tuples come
  back as lists, nested int keys come back as strings, Enums come back as raw values. Because
  the same normalization is applied when hashing on both sides, the load-time hash check
  **passes**, so the corruption is invisible — replay code comparing `params["shape"]`
  against a tuple, or re-keying a nested dict by int, silently misbehaves.
- **Repro:**
  ```python
  st = CompressionState(..., compression_params={"topk": (1, 2)})
  rec.record(0, "compression", [st]); rec.load(0, "compression")[0].compression_params
  # -> {'topk': [1, 2]}            (tuple silently became list)
  # loaded.content_hash == hash_model(st) is still True
  # nested keys: {"nested": {1: "x"}} -> {"nested": {"1": "x"}}; hash_model equal to str-keyed twin
  ```
- **Suggested fix:** Either reject non-JSON-native values in `Any` fields at record time
  (fail loud like the NaN path does), or document the normalization. If fidelity matters,
  type-tag them (e.g. `{"__falcon_tuple__": [...]}`) and restore on load. Add a round-trip
  test with a tuple in `compression_params`.

### M3. `client_id` case collision silently destroys one client's recorded state (Windows/macOS)

- **Where:** `falcon/recorder/recorder.py:86-94` (duplicate check is a case-sensitive `set`),
  filenames built from raw `client_id` at `recorder.py:94`.
- **What breaks:** Two clients whose ids differ only in case map to the same file on
  case-insensitive filesystems (this dev machine is Windows; CI may be Linux — so this can
  pass CI and corrupt locally, or vice versa). The second write overwrites the first; `load`
  returns fewer clients than recorded, with **no error**. Same class of bug: ids differing by
  trailing dots/spaces (`"c0"` vs `"c0 "`), which Win32 strips.
- **Repro (executed on this machine):**
  ```python
  rec.record(0, "local", [mklocal("c0", 1.0), mklocal("C0", 9.0)])
  # files on disk: ['c0.json', 'c0.npz']   <- only ONE pair
  rec.load(0, "local")  # -> [ClientLocalState(client_id='C0', update=[9.0])]  — c0 gone, no error
  ```
- **Suggested fix:** Normalize before duplicate-checking (`client_id.casefold()` in `seen`),
  or hash/percent-encode `client_id` for the filename and keep the real id inside the JSON.
  At minimum, on `os.name == "nt"` reject ids that casefold-collide.

### M4. The test suite cannot detect a broken hash — constant-hash mutant passes everything

- **Where:** `tests/unit/test_recorder.py`, `tests/replay/test_clean_replay.py` (absence of
  any sensitivity assertion).
- **What breaks:** Every hash assertion in the suite compares *equal states for equality*;
  nothing asserts that *different states hash differently*. I replaced `hash_model` with
  `lambda m: "0"*64` (monkeypatched into `falcon.recorder.hashing`, `falcon.recorder.recorder`,
  and the test module) and ran the real test functions:
  - `test_record_load_round_trip_for_every_stage_state` — **passes**
  - `test_content_hash_is_stable_across_saves` — **passes**
  - `test_same_seed_produces_equal_recorded_stage_hashes` — **passes**

  A recorder that emits a constant `content_hash` for every state — making
  `tests/replay/test_clean_replay.py` (the Contract §5 gate) meaningless — would ship green.
- **Suggested fix:** Add assertions that (a) two states differing by one array ulp / one scalar
  / one dict entry have different `content_hash`; (b) a different seed in
  `test_clean_replay` yields different `stage_hashes`; (c) tampering with a written `.json`
  byte makes `load` raise (`recorder.py:192-193` is currently uncovered).

---

## MINOR

### m1. Reserved key `__falcon_array__` in user data makes a recorded state unloadable

- **Where:** `falcon/recorder/hashing.py:13` (marker), `falcon/recorder/recorder.py:199-204`
  (any dict containing the marker key is treated as an array reference).
- **What breaks:** `dict[str, Any]` fields are user data. If they contain the key
  `__falcon_array__` (plausible in `compression_params` for an algorithm that logs internals),
  the state records without complaint but `load` fails: `ValueError: missing recorded array`
  for a dangling key, or `content hash mismatch` / pydantic `ValidationError` when the key
  accidentally names a real array (the array gets substituted into the wrong field first).
  Fails loud, not silent — but the failure is far removed from the cause.
- **Fix:** Namespace the marker (e.g. include a fixed magic salt and require `"dtype"`,
  `"shape"`, `"path"` all present before treating as a reference), or reject the key at record
  time with a clear message.

### m2. Non-int `round_id` is recorded but invisible to `stage_hashes`

- **Where:** `falcon/recorder/recorder.py:73` (no validation) vs `recorder.py:124-128`
  (`stage_hashes` skips directories whose suffix doesn't parse as `int`).
- **What breaks:** `rec.record(1.5, "evaluation", st)` succeeds and `load(1.5, ...)` works,
  but `stage_hashes()` returns `{}` — the recorded boundary silently drops out of whole-run
  comparison. Verified. Negative ids do work. Caller bug, but a silent one in a tool whose
  entire purpose is comparison.
- **Fix:** `if not isinstance(round_id, int) or isinstance(round_id, bool): raise TypeError`
  in `record`/`load`.

### m3. Stale `Generator` references survive `load_state_dict`

- **Where:** `falcon/replay/rng.py:62-65` — `_streams.clear()` drops the registry's objects,
  but generators already handed out stay alive with pre-restore state.
- **What breaks:** Replay code that cached `gen = rng.stream("aggregation")` before an
  intervention/restore keeps drawing from the *old* sequence after `load_state_dict`,
  silently breaking bit-exact replay. Verified (old reference still yields values post-load).
- **Fix:** Document loudly in `load_state_dict` / class docstring ("re-fetch all streams after
  calling this"), or hand out lightweight proxies that always go through the registry.

### m4. `stage_hashes` value depends on representation (single model vs 1-element list)

- **Where:** `falcon/recorder/recorder.py:136-143` — a per-client stage recorded as a list
  hashes `sha256(json([content_hash,...]))`, while the same stage recorded as a bare model
  hashes to the raw `content_hash`. Identical logical content, different boundary hash.
  Note: the working-tree update to Contract §1 now mandates list-recording for per-client
  stages ("never once per client"), but the recorder does not enforce it.
- **Fix:** Route the single-model case through the same one-element-list canonicalization
  for per-client stages, or reject bare models for `"local"`/`"compression"`.

### m5. Contract/task drift (docs-level)

- Contract §4 specifies `Recorder.load(run_id, round_id, stage)`; T1 and the code implement
  `load(round_id, stage)` with `run_id` bound at construction. Followed T1, so code is fine —
  but `docs/CONTRACTS.md:69` should be updated or it will mislead the T2 implementer.
- T1 asked for the replay placeholder to be "marked"; `tests/replay/test_clean_replay.py`
  carries no pytest mark.

---

## Weak tests (would pass even if the code were wrong)

1. **No hash-sensitivity test anywhere** — see M4 (mutant proof). Highest-impact gap.
2. **`hash_array` is never tested** — not imported by any test; M1's collision would never
   be caught.
3. `tests/unit/test_recorder.py` exercises only 1-D contiguous float64 arrays and str-keyed
   dicts, so the M2 normalizations (tuple, nested int keys, Enum) sail through. No negative
   tests: unknown stage, list passed to a non-per-client stage, duplicate `client_id`,
   traversal `client_id` (`"../x"`), or the tamper-detection path (`recorder.py:192`).
4. `tests/replay/test_clean_replay.py` runs both "runs" in one process, so it cannot catch
   cross-process/global-state nondeterminism, and never asserts that a *different* seed
   produces *different* hashes. (Acceptable as a T1 placeholder per the task, but the T2
   successor must fix both.)
5. `tests/unit/test_rng.py` does not test snapshot independence (mutating the dict returned
   by `state_dict()` must not corrupt the live registry — the `deepcopy` at `rng.py:46` is
   untested), nor `load_state_dict` error paths, nor the m3 stale-reference caveat.

---

## Verified solid (attacked, did not break)

- **RNG order independence:** 50 streams created in reverse order produce identical sequences.
  Seeding uses the *full* 256-bit sha256 digest as `spawn_key` (`rng.py:30`) — no truncation;
  name-collision risk is a sha256 collision, negligible.
- **`state_dict`/`load_state_dict`:** bit-exact mid-sequence restore, including through a JSON
  round-trip (huge PCG64 ints survive), and streams created lazily *after* a restore match the
  original run (root_seed is carried in the snapshot). Registry is cleared on load; garbage
  state dicts, negative seeds, and empty stream names all raise `ValueError` loudly.
- **`content_hash` written-after-hashing is consistent:** the field is excluded from the hash
  (`hashing.py:39`), re-added to the JSON, and re-verified on load — the hinted
  write-order mismatch does not exist. Verified end-to-end.
- **Hash canonicalization:** dict insertion order normalized; 1-ulp differences in arrays *and*
  scalars change the hash; `-0.0` vs `0.0` distinguished; unicode via `ensure_ascii=False` +
  UTF-8 is deterministic; nested models/arrays hashed in fixed field order.
- **`.npz` byte determinism:** numpy 2.5.1 writes fixed 1980-01-01 zip timestamps — two saves
  of identical arrays are byte-identical (my wall-clock suspicion was refuted).
- **Path traversal:** `_safe_component` rejects `/ \ :`, `"."`, `".."`, empty and non-str for
  both `run_id` and `client_id`; duplicate exact ids rejected.
- **Refuted hypotheses (tested, no bug on this box):** Windows reserved device names
  (`CON`, `NUL`) as run/client ids worked fine; dot-prefixed client ids (`".c0"`) glob and
  load fine on Python 3.13.11; F-order arrays round-trip with flags and values intact;
  re-recording a stage with fewer clients cleans up stale files correctly.

No source or test files were modified; all probe scripts were deleted after use.
