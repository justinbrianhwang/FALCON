# Adversarial Review — T3 (matcher), T6 (metrics/attribution), T7 (reporting)

Reviewer: Kimi (round 2, adversarial)
Scope: `falcon/matcher/`, `falcon/metrics/effects.py`, `falcon/attribution/analyzer.py`, `falcon/reporting/`, and their unit/integration tests.
Method: spec cross-check against Plan.md §11.3, §12, §14, §15; throwaway repro scripts (deleted after use); a 7-mutant sweep run against the 5 target test files in a copied tree (no source or test file was modified). Baseline: all 25 target tests pass.

---

## CRITICAL

### C1. Earliest-decisive-stage rule (§15.2) is disabled on the only production path — downstream restoration trap is live

- **Where:** `falcon/attribution/analyzer.py:35` (`epsilon_tie: float = 1e-9`), `analyzer.py:136-138` (first-divergence promotion gated by `epsilon_tie`), `falcon/reporting/analyze.py:95-104` (`attribute()` called **without** `epsilon_tie`; no CLI flag exists).
- **Failure scenario:** §15.2 says FALCON must not pick the max-SRE stage because a downstream state can encode all upstream damage. The defense is the promotion at line 136, but it only fires when the true origin's score is within `epsilon_tie` of the max. With the default `1e-9` — the value `analyze_pair` and the CLI always use — a downstream stage that restores even slightly better than the true origin is named the unique `origin_candidate`, with no note. The only test that exercises the trap (`tests/unit/test_analyzer.py:84-99`) passes `epsilon_tie=0.1` by hand, a value no caller in the shipped path can set.
- **Repro (verified):** pair with `first_divergence_stage="local"`, restore/inject giving `BIS(local)=0.90`, `BIS(aggregation)=0.95`, default `epsilon_tie`:
  `origin_ranking == ["aggregation", "local"]`, `roles == {"aggregation": "origin_candidate", ...}`, `notes == []`. The true origin `local` is demoted to `carrier_or_amplifier`.
- **Suggested fix:** separate the two conflated tolerances: keep `epsilon_tie=1e-9` for float ties, add a `decisive_margin` (e.g. 0.05) for "interventionally indistinguishable effects", and have the promotion use the margin. Thread it through `analyze_pair` and the CLI (`--decisive-margin`). Alternatively rank lexicographically per §15.2: among stages whose scores are within the margin of the max, earliest divergent stage wins.

### C2. Ambiguous evidence still fabricates a unique origin — §15.3 violated in report and verdict

- **Where:** `falcon/attribution/analyzer.py:140-141` (UNRESOLVED is only a note string), `analyzer.py:143-153` (top stage still gets role `origin_candidate`), `falcon/reporting/report.py:87-93` (counterfactual explanation generated from `origin_ranking[0]` regardless), `report.py:117-127` (benchmark verdict is a hard yes/no on `origin_ranking[0]`).
- **Failure scenario:** Plan §15.3: "Forcing a unique answer in underidentified cases would be scientifically incorrect." The schema has no unresolved/origin-set outcome, so: (a) on a tie, the analyzer emits `UNRESOLVED_BETWEEN:local,compression` yet still ranks `local` first as `origin_candidate`; (b) `render_markdown` prints the definitive sentence "Restoring local closes 80.0% of the gap"; (c) in benchmark mode the report scores the tie as `Prediction matches injected stage: **no**` — a forced wrong answer where the correct output is "unresolved". A related edge: when **all** stages have ~zero effect, first-divergence promotion still crowns the first-divergent stage `origin_candidate` (verified: `nSRE=0.0` stage named origin).
- **Repro (verified):** `_result("local","restore",0.82)` + `_result("compression","inject",0.58)` with ground truth `compression` → notes contain `UNRESOLVED_BETWEEN:local,compression`, but the markdown says `Restoring local closes 80.0% of the gap` and `Prediction matches injected stage: **no**`.
- **Suggested fix:** add an explicit `outcome` field to `AttributionReport` (`unique_origin` / `origin_set` / `unresolved_between` / `insufficient_failure_gap` / `invalid_intervention`) per §15.3. When unresolved: suppress the `origin_candidate` role, have `render_markdown` print the tied set instead of a single-stage counterfactual, and score the benchmark verdict as "unresolved", not "no".

### C3. Gap guard admits negative and NaN gaps — sign-flipped or NaN effects yield fabricated origins

- **Where:** `falcon/attribution/analyzer.py:110` (`if abs(gap) < min_gap:`), same pattern in `falcon/metrics/effects.py:35` (`nsre`) and `effects.py:58` (`nsie`).
- **Failure scenario:** Plan §14.1 defines the experiment as meaningful only when `G > τ_M` (strictly positive, predeclared). The `abs()` guard lets two degenerate cases through:
  1. **Negative gap** (the "failure" run is *better* than the reference — happens with benign-drift failures or noise): normalization divides by a negative G, so a restore that **worsens** the outcome (SRE = −0.05) gets a **positive** nSRE (+0.125) and the stage becomes `origin_candidate`. Verified repro: `m_ref=0.5, m_fail=0.9` → `ranking == ["local"]`, `roles == {"local": "origin_candidate"}`, `notes == []`.
  2. **NaN gap** (diverged training → `loss=NaN`, the most common real FL failure): `abs(nan) < min_gap` is `False`, so analysis "proceeds"; every nSRE is NaN; `sorted()` with NaN keys silently keeps pipeline order; the first intervenable stage becomes `origin_candidate` with zero notes, and the report prints `Restoring local closes nan% of the gap`. Verified.
- **Suggested fix:** `if not math.isfinite(gap) or gap < min_gap:` → `INSUFFICIENT_FAILURE_GAP` (or a distinct `NONPOSITIVE_FAILURE_GAP` note). Apply the same finiteness/positivity check in `effects.nsre/nsie`, and reject NaN values in `InterventionResult.outcome_metrics[metric]` at grouping time (`analyzer.py:45`).

---

## MAJOR

### M1. Hash-coverage gap: a boundary with a missing `.json` sidecar is invisible — tampered pre-failure state passes as clean MATCHED

- **Where:** `falcon/matcher/matcher.py:45` (`_boundaries`: boundary exists iff `stage.json` sidecar is a file **or** `stage/` is a dir), `matcher.py:50-66` (fingerprint reads whatever the sidecar points at; orphan `.npz` files are never seen).
- **Failure scenario:** if `round_0/aggregation.json` is missing in **both** runs (partial deletion/corruption), the `(0, aggregation)` boundary vanishes from both boundary sets, so `stage_hash_coverage` passes and the round-0 aggregate is **never hashed or compared**. Verified: delete both sidecars, then add `+100.0` to the failure run's round-0 `aggregation.npz` (a genuine pre-failure divergence) → `status == "MATCHED"`, all checks pass, zero warnings, first divergence reported as `(1, local)`. An INVALID pair slips through as MATCHED. Root cause: the matcher re-implements file discovery instead of trusting the recorder's own load+verify path (`Recorder._read_model` verifies content hashes and would have raised).
- **Suggested fix:** treat `.npz`-without-`.json` (and vice versa) as a corrupt boundary → `INVALID_PAIR`; consider fingerprinting via `Recorder.load` + `hash_model` so structural corruption fails closed instead of being skipped.

### M2. Matcher crashes on a truncated `.npz` instead of returning INVALID_PAIR

- **Where:** `falcon/matcher/matcher.py:176` — the guard around `_stage_hashes` catches `(OSError, ValueError, KeyError, TypeError)` but `np.load` on a truncated archive raises `zipfile.BadZipFile` (a direct `Exception` subclass).
- **Repro (verified):** truncate `round_0/aggregation.npz` to half its size → `validate_pair` propagates `zipfile.BadZipFile: File is not a zip file`; the CLI dies with a traceback. (Random garbage bytes do raise `ValueError` and are caught — only the valid-zip-header case escapes.)
- **Suggested fix:** add `zipfile.BadZipFile` (or broaden to `Exception`) to the `hashes_loadable` guard; a validator must fail closed, never crash.

### M3. `code_version` / software environment is never compared (§11.3 check missing)

- **Where:** `falcon/matcher/matcher.py:153-164` — checks cover seed, rounds, dataset, config delta; `RunMetadata.code_version` (`falcon/schema/states.py:113`) is compared nowhere. Plan §11.3 explicitly lists "software environment" among matcher checks.
- **Repro (verified):** identical pair, `code_version="git:aaa"` vs `"git:bbb"` → `status == "MATCHED"`, all checks pass. Runs recorded by different code can be silently matched even when the recorded states happen to align.
- **Suggested fix:** add a `same_code_version` check (mismatch → at minimum a warning, arguably INVALID for causal estimates per §11.3).

### M4. Failures active from round 0 make both pre-failure checks vacuously true

- **Where:** `falcon/matcher/matcher.py:181-190` (`pre_failure = {b for b in ... if b[0] < start_round}` is empty when `active_rounds[0] == 0`; `all([])` is `True`), `matcher.py:192-196` (`range(min(start_round, ...))` is empty → selection check also vacuous).
- **Failure scenario:** for a failure starting at round 0, "MATCHED" asserts pre-failure equivalence that was never verified on any recorded state — the pair's comparability rests solely on config/seed equality. Verified: failure `active_rounds=(0,1)` stage `local`, failure run's **round-0 selection tampered** to a different client set → `pre_failure_hashes_match == True`, `selection_matches_pre_failure == True`; only the first-divergence-stage heuristic saves it to `MATCHED_WITH_WARNINGS`, and had the tamper been at the configured stage it would be clean MATCHED. Note `analyze_pair` accepts `MATCHED_WITH_WARNINGS` for causal estimates.
- **Suggested fix:** when the pre-failure boundary set is empty, emit an explicit warning ("no pre-failure boundaries recorded — match rests on config/seed only") so the status is at least `MATCHED_WITH_WARNINGS`.

### M5. Multi-round intervention results silently overwrite each other in the analyzer

- **Where:** `falcon/attribution/analyzer.py:46` (`grouped.setdefault(stage, {})[mode] = result` — keyed by `(stage, mode)` only) vs `falcon/reporting/analyze.py:84-87` (one result per `(round, stage, mode)` when `rounds` is a list).
- **Failure scenario:** call `analyze_pair(..., rounds=[1, 2])` → 24 results; per stage/mode only the **last round's** result survives, with no note. Verified at unit level: restore results `0.55` (round 1) and `0.89` (round 2) for `local` → reported `nSRE == 0.975` (round 2), round-1 evidence gone silently. Which round wins depends on list order.
- **Suggested fix:** key `grouped` by `(stage, mode, round)` and aggregate explicitly (e.g. mean nSRE per stage, per §14.6/§21), or reject duplicate `(stage, mode)` results with an `INVALID_INTERVENTION`-style note.

### M6. Sham gate passes vacuously when sham controls are missing or invalid; shams only ever target the failure run

- **Where:** `falcon/attribution/analyzer.py:43-46` (invalid results excluded from `grouped`), `analyzer.py:114-119` (gate reads `stage_effects.get(stage, {}).get("sham_dev", 0.0)` — absent sham ⇒ 0.0 ⇒ pass), `falcon/reporting/analyze.py:76-80` (`target_run_id` is the failure run for every mode except `inject`, so reference-run shams are never executed; the `m_ref` baseline branch of `_sham_deviation` is dead code).
- **Failure scenario:** §12.4 ("a method that reports large sham effects is invalid") and §15.2 criterion 4 require sham controls. Verified: (a) a broken sham (`valid=False`) yields only an `INVALID_INTERVENTION` note and the ranking proceeds; (b) no sham results at all → ranking proceeds with **zero** notes. There is no "control absent" signal, so a report can claim sham-validated origins with no sham evidence.
- **Suggested fix:** emit a `SHAM_CONTROL_MISSING:<stage>` note when a ranked stage lacks a valid sham result; consider refusing a clean ranking when no valid sham exists at all; run shams on the reference run too (or delete the dead baseline branch).

### M7. Weak tests — six mutants survive the suite

Mutation sweep (7 mutants applied to a copied tree, all 25 target tests run per mutant):

| Mutant | Result | Why it matters |
|---|---|---|
| `pre_failure` window `<` → `<=` (`matcher.py:185`) | **killed** (3 failed) | good |
| `effects.bis`: `abs(nSRE−nSIE)` → `(nSRE−nSIE)` | SURVIVED | `tests/unit/test_effects.py:63-67` only exercises `nSRE > nSIE`; penalty direction for `nSRE < nSIE` unpinned |
| `analyzer.py:110`: `abs(gap)` → `gap` | SURVIVED | no negative-gap test anywhere (twin of C3) |
| `matcher.py:113`: drop `parsed_failure == failure.failure` | SURVIVED | config-vs-metadata failure-spec inconsistency unpinned (behavior verified correct, but untested) |
| `matcher.py:110`: drop `reference.failure is None` | SURVIVED | "reference carries its own failure spec" unpinned (behavior verified correct, but untested) |
| `analyzer.py:136-138`: delete first-divergence promotion | SURVIVED | see below |
| `analyzer.py:118`: sham gate `>` → `>=` | SURVIVED | tolerance boundary unpinned (benign but tells) |

The `promotion_removed` survivor is the serious one: `test_first_divergence_wins_downstream_restoration_trap` ties `local` vs `aggregation` within `epsilon_tie=0.1`, where the tie-block re-sort by pipeline order (`analyzer.py:128-133`) accidentally produces the same winner as the promotion. The two mechanisms diverge when an **upstream, non-divergent** stage ties the first-divergent one — verified: with `first_divergence_stage="local"` and `selection` tied with `local`, real code correctly promotes `local`, the mutant wrongly picks `selection`, and no test can tell. Add that case as a regression test together with the C1 fix.

---

## MINOR

1. **Matcher CLI exits 0 on INVALID_PAIR** — `falcon/matcher/__main__.py:39` returns 0 unconditionally; the reporting CLI (`falcon/reporting/__main__.py:54`) returns 1. A CI gate on the matcher exit code would proceed with an invalid pair. Verified. Return 1 on `INVALID_PAIR`.
2. **Overshoot percentages printed without the "requires analysis" flag §14.2 asks for** — gap `0.006` barely over `min_gap=0.005` gives `nSRE=66.7` and the report prints `Restoring local closes 6666.7% of the gap` with no note (`report.py:14-15, 90-93`). Values not clipped (correct), but never flagged either.
3. **BIS degenerates at the default λ** — `effects.py:63-73` with `lam=0.5` is algebraically `min(nSRE, nSIE)`; the mean term cancels. Harmless but worth a comment, since the test pins `0.6` for exactly this degenerate case.
4. **Mixed evidence scales in one ranking** — `analyzer.py:82-89` ranks BIS (two-way evidence) against bare nSRE/nSIE (one-way evidence) on the same axis; a restore-only stage at 0.95 beats a bidirectionally-confirmed stage at 0.90. Consider ranking bidirectional stages ahead of single-evidence stages at equal magnitude.
5. **SAE (§14.5) is never computed** — `effects.sham_adjusted` has no caller; the analyzer gates on sham deviation but never reports `SRE − SRE_sham`.
6. **Roles don't implement §14.9** — no standardized deviation `D_j`; `bystander` is assigned from the raw score `< 0.1` (`analyzer.py:150-151`), not from "significant state deviation + negligible sham-adjusted effect". Fine for MVP, but the role names already promise the §14.9 semantics.
7. **No warning when first divergence falls outside the failure's active window** — `matcher.py:214-222` warns on stage mismatch only; a first divergence at round 3 for `active_rounds=(0,0)` with a matching stage passes clean.
8. **`analyze_pair` crashes instead of degrading on missing data** — `analyze.py:89-100`: absent terminal `evaluation` → uncaught `FileNotFoundError`; absent metric key → uncaught `KeyError`. Also, corrupt recorded states make `Recorder.load` raise `ValueError` through `apply_intervention` (`engine.py` catches only `_InvalidIntervention`), aborting the whole analysis.
9. **UNRESOLVED note names only the top two** — `analyzer.py:140-141`; a 3-way tie reports `UNRESOLVED_BETWEEN:A,B` and hides C.

---

## What held up (attacked, survived)

- **Metric signs/direction (§14.1–14.3):** `failure_gap`, `sre`, `sie` correct for higher- and lower-is-better; hand-computed normalized values match; `nSRE > 1` and negative nSRE preserved unclipped per §14.2; `min_gap` guard refuses tiny `|G|`; BIS formula matches §14.4 literally.
- **Matcher core soundness:** seed/rounds/dataset mismatches caught; nested config delta (deep dict equality) caught; config-failure vs metadata-failure inconsistency caught (verified INVALID); a reference run carrying its own failure spec caught; pre-failure hash tampering caught with correct first-divergence `(0, evaluation)`; identical runs downgraded to `MATCHED_WITH_WARNINGS` with the right warning; boundary-set asymmetry fails `stage_hash_coverage`; garbage-byte `.npz` → INVALID (not crash); the `<`-window mutant was killed by tests.
- **Analyzer gates:** `INVALID_PAIR` short-circuits before any intervention (`analyze.py:50-61`, verified end-to-end via seed tamper); small `|G|` → `INSUFFICIENT_FAILURE_GAP` with no ranking; a present, exceeding sham deviation kills the whole report; invalid interventions are excluded and named in notes; deterministic-replay sham deviation is exactly 0.0 end-to-end.
- **Reporting separation (§11.7):** measured evidence, inferred roles, and ground truth are in separate, labeled sections; ground truth only appears in its own `(benchmark)` section and is sourced from the failure run's metadata; pair checks and invalid interventions are tabled; reporting CLI exits 1 on INVALID_PAIR; the full end-to-end pipeline attributes a real `aggressive_topk` compression failure to the correct stage with 100%/100% restore/inject.

## Bottom line

The metric math and the matcher's happy-path checks are solid. The dangerous cluster is in attribution semantics: the two guards that carry the scientific claims — §15.2 earliest-decisive-stage (C1) and §15.3 ambiguity handling (C2) — are effectively inert on the shipped path, and the gap guard's `abs()` (C3) admits exactly the degenerate runs (improved/NaN metrics) a failure-debugging tool will meet most. All three fabricate a confident unique origin from evidence that does not support one. On the matcher side, the fail-open boundary discovery (M1) and the uncaught `BadZipFile` (M2) should fail closed, and the missing `code_version` comparison (M3) is a straight §11.3 gap.
