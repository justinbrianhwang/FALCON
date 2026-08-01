# Task T8 — Passive/terminal localization baselines

**Assignee:** Kimi
**Contract:** Plan.md §19.1–19.2, §35 Task 7. You own `falcon/baselines/` (new package); read-only elsewhere. These are the rivals FALCON must beat in E1/E2 — implement them fairly, not as strawmen.

## Deliverables

### 1. `falcon/baselines/passive.py` — passive stage-anomaly localization (Plan §19.2)

`passive_stage_scores(ref_root: Path, fail_root: Path, reference_run_id: str, failure_run_id: str) -> dict[str, float]`

For each intervenable stage, compute a stage-anomaly score from RECORDED states only (no replay, no interventions):

- selection: Jaccard distance between selected_ids per round, averaged;
- local: mean over rounds/clients of relative L2 deviation of updates (match by client id; unmatched clients count as max deviation 1.0);
- compression: same on post-compression updates;
- aggregation: relative L2 deviation of the aggregate vector per round, averaged.

`passive_localize(scores) -> str` — argmax stage. Deterministic tie-break: STAGES order.

### 2. `falcon/baselines/terminal.py` — terminal-only features (Plan §19.1)

`terminal_features(run_root: Path, run_id: str) -> np.ndarray` — fixed-order vector from the final OutcomeState: final accuracy, final loss, per-class accuracies (sorted keys), accuracy slope over last 3 rounds, plus global-update norm of the last round's aggregate. Document the feature order.

`NearestCentroidStageClassifier` — `fit(X: list[np.ndarray], y: list[str])`, `predict(x) -> str`: nearest centroid in feature space (euclidean, z-normalized per feature using fit statistics; guard zero variance). This is the weakest §19.1 baseline; sklearn models come later if ever needed — do NOT add sklearn.

### 3. Tests: `tests/unit/test_baselines.py`

- passive: build two tiny recorded runs (real Recorder) where exactly one stage's states differ → that stage argmax; tie-break determinism; unmatched-client handling.
- terminal: feature vector shape/order stability; centroid classifier recovers labels on separated synthetic clusters; zero-variance feature guard.
- One integration-style test: reference vs T4 aggressive_topk failure run → passive_localize returns "compression" (an honest baseline win on an easy case).

## Rules

numpy/pydantic/stdlib only. Do not touch falcon/reporting (other developer, concurrent). Full suite green. No git commit.
