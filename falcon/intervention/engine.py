"""Intervention engine: restore / inject / sham (Task T5, Plan §11.5, §13.1–13.3).

Replay strategy (PM decision): full deterministic re-execution with an
overlay — re-run ``run(cfg)`` from round 0 and, at the intervention boundary,
swap the produced stage state for the recorded source state, then continue
downstream. No checkpoint restoration.

Round windows (T13, Plan §13.5): ``InterventionSpecification.round_window =
[t1, t2]`` turns the single boundary into one replay in which the stage state
is replaced at EVERY round of the inclusive window (source = the matched
run's recorded state for the same round; ``round_id`` is ignored). Validation
runs per round exactly as in the single-round case; any invalid round rejects
the whole window with ``reason`` ending in ``:<round>`` — no partial windows.
A windowed sham round-trips the LIVE state at every window round under the
same drift gate.

Sham design (T5-F finding 2): a sham is evidence about the intervention
machinery, not a repair tool. It (a) runs a NO-overlay replay of the target
and requires every recorded boundary hash to match — any drift invalidates
the sham (``replay_drift:<round>/<stage>``); (b) overlays the LIVE boundary
state after a Recorder save/load round-trip (a pure serialization test),
never the recorded target state, which could silently erase drift; and
(c) for the evaluation stage compares the RECOMPUTED outcome against the
recording, since comparing the self-replaced outcome is tautological.

Cross-run gates (T5-F findings 3–4): restore/inject require matched run
metadata (same seed, equal dataset config, config delta limited to
``failure``) and matching lineage hashes at the replaced boundary —
violations are ``valid=False``, never warnings.

``runs_root`` is the recorder root directory: recorded runs live at
``runs_root/runs/<run_id>/`` (``metadata.json`` + per-round stage files), the
same layout ``falcon.recorder.recorder.Recorder(root_dir, run_id)`` writes.
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    STAGES,
    AggregationState,
    ClientLocalState,
    CompressionState,
    InterventionResult,
    InterventionSpecification,
    OutcomeState,
    RunConfig,
    RunMetadata,
    SelectionState,
)

#: per-client stages recorded once per stage as a list (CONTRACTS §1)
_LIST_STAGES = ("local", "compression")
#: lineage hash field per list stage, checked against the live replay
_LINEAGE_FIELD = {"local": "base_model_hash", "compression": "uncompressed_hash"}
#: array field per stage, checked against the live computed state
_ARRAY_FIELD = {"local": "update", "compression": "update", "aggregation": "aggregate"}
#: pydantic type a recorded boundary must load as, per stage
_EXPECTED_STATE_TYPE = {
    "selection": SelectionState,
    "local": ClientLocalState,
    "compression": CompressionState,
    "aggregation": AggregationState,
    "evaluation": OutcomeState,
}
#: recorder/serialization failures translated into ``valid=False`` reasons
_LOAD_ERRORS = (ValueError, TypeError, OSError, EOFError, zipfile.BadZipFile)


class _InvalidIntervention(Exception):
    """Internal abort signal: a validation check failed at the live boundary.

    Raised by the overlay during re-execution and caught by
    ``apply_intervention`` — validation failures must surface as
    ``InterventionResult(valid=False, reason=...)``, never as exceptions.
    """


class _ReplacementOverlay:
    """Runner overlay swapping boundary states for recorded replacements.

    Holds one replacement per intervention round (a single-round spec is a
    one-entry window). Validation that needs the live computed state happens
    here, at each boundary: array shape/dtype/finiteness (vs the live state),
    client identity, scoped client_ids presence in BOTH live and source, and
    the lineage-hash check. A lineage mismatch means the replacement descends
    from a different base model — transplanting its delta is not a
    restoration of the reference stage state, so it is REJECTED with
    ``lineage_mismatch`` (T5-F finding 3), never downgraded to a warning.

    ``tag_round`` (window mode, T13): any per-round validation failure is
    re-raised with ``:<round>`` appended, so the reason names the window
    round that rejected — a window is all-or-nothing, never partial.
    """

    def __init__(
        self,
        replacements: dict[int, object],
        stage: str,
        client_ids: list[str] | None,
        tag_round: bool = False,
    ):
        self._replacements = replacements  # round_id -> recorded source state
        self._stage = stage
        self._client_ids = client_ids  # None => whole-stage replacement
        self._tag_round = tag_round
        self.fired = 0

    def override(self, round_id: int, stage: str, state):
        if round_id not in self._replacements or stage != self._stage:
            return state
        self.fired += 1
        replacement = self._replacements[round_id]
        try:
            if stage in _LIST_STAGES:
                return self._override_list(state, replacement, round_id)
            return self._override_single(state, replacement, round_id)
        except _InvalidIntervention as exc:
            if self._tag_round:
                raise _InvalidIntervention(f"{exc}:{round_id}") from None
            raise

    # --- single states: selection / aggregation / evaluation ---

    def _override_single(self, live, replacement, round_id: int):
        array_field = _ARRAY_FIELD.get(self._stage)
        if array_field is not None:
            self._check_array(
                getattr(replacement, array_field),
                getattr(live, array_field),
                round_id,
                context=f"stage {self._stage!r}",
            )
        return replacement

    # --- list states: local / compression ---

    def _override_list(self, live: list, source: list, round_id: int) -> list:
        live_by_id = {state.client_id: state for state in live}
        source_by_id = {state.client_id: state for state in source}

        if self._client_ids is None:
            live_ids = [state.client_id for state in live]
            source_ids = [state.client_id for state in source]
            if source_ids != live_ids:
                raise _InvalidIntervention(
                    f"client_mismatch: source client_ids {source_ids} do not "
                    f"match live client_ids {live_ids} at round "
                    f"{round_id} stage {self._stage!r}"
                )
            replaced_ids = live_ids
        else:
            missing = [
                cid
                for cid in self._client_ids
                if cid not in live_by_id or cid not in source_by_id
            ]
            if missing:
                raise _InvalidIntervention(
                    f"scoped_clients_missing: client_ids {missing} not present "
                    f"in BOTH live and source state at round {round_id} "
                    f"stage {self._stage!r}"
                )
            replaced_ids = self._client_ids

        for cid in replaced_ids:
            live_entry = live_by_id[cid]
            source_entry = source_by_id[cid]
            self._check_array(
                getattr(source_entry, _ARRAY_FIELD[self._stage]),
                getattr(live_entry, _ARRAY_FIELD[self._stage]),
                round_id,
                context=f"stage {self._stage!r} client {cid!r}",
            )
            lineage_field = _LINEAGE_FIELD[self._stage]
            if getattr(source_entry, lineage_field) != getattr(live_entry, lineage_field):
                # the replacement descends from a different base model than
                # the live replay: a delta trained/compressed under another
                # lineage is not a restoration of this stage state (finding 3)
                raise _InvalidIntervention(
                    f"lineage_mismatch: {lineage_field} of client {cid!r} does "
                    f"not match the live replay at round {round_id} "
                    f"stage {self._stage!r}"
                )

        if self._client_ids is None:
            return list(source)
        replaced = set(self._client_ids)
        return [
            source_by_id[state.client_id] if state.client_id in replaced else state
            for state in live
        ]

    def _check_array(self, source_array, live_array, round_id: int, context: str) -> None:
        source = np.asarray(source_array)
        live = np.asarray(live_array)
        if source.shape != live.shape:
            raise _InvalidIntervention(
                f"shape_mismatch: source shape {tuple(source.shape)} != live "
                f"shape {tuple(live.shape)} at round {round_id} {context}"
            )
        if source.dtype != live.dtype:
            raise _InvalidIntervention(
                f"dtype_mismatch: source dtype {source.dtype} != live dtype "
                f"{live.dtype} at round {round_id} {context}"
            )
        if not np.all(np.isfinite(source)):
            raise _InvalidIntervention(
                f"non_finite_state: source array contains non-finite values "
                f"at round {round_id} {context}"
            )


class _ShamOverlay:
    """Sham replacement: the LIVE boundary after a Recorder round-trip.

    The recorded target state is NEVER overlaid (T5-F finding 2): swapping in
    recorded content would repair replay drift at the intervention boundary
    and make ``sham_deviation_* == 0`` tautological. Round-tripping the live
    state tests only serialization fidelity. ``live_states`` captures the
    pre-overlay boundary per intervention round so an evaluation-stage sham
    can compare the RECOMPUTED outcome against the recording instead of the
    self-replaced one. With a round window (T13) the round-trip happens at
    EVERY round in the window, under the same drift gate.
    """

    def __init__(self, rounds, stage: str, client_ids: list[str] | None):
        self._rounds = frozenset(rounds)
        self._stage = stage
        self._client_ids = client_ids  # None => whole-stage round-trip
        self.fired = 0
        self.live_states: dict[int, object] = {}

    def override(self, round_id: int, stage: str, state):
        if round_id not in self._rounds or stage != self._stage:
            return state
        self.fired += 1
        self.live_states[round_id] = state
        if stage in _LIST_STAGES:
            return self._round_trip_list(state, round_id)
        return _round_trip_through_serialization(state, round_id, stage)

    def _round_trip_list(self, live: list, round_id: int) -> list:
        if self._client_ids is not None:
            live_ids = {state.client_id for state in live}
            missing = [cid for cid in self._client_ids if cid not in live_ids]
            if missing:
                raise _InvalidIntervention(
                    f"scoped_clients_missing: client_ids {missing} not present "
                    f"in the live state at round {round_id} "
                    f"stage {self._stage!r}"
                )
        tripped = _round_trip_through_serialization(live, round_id, self._stage)
        if self._client_ids is None:
            return tripped
        tripped_by_id = {state.client_id: state for state in tripped}
        keep = set(self._client_ids)
        return [
            tripped_by_id[state.client_id] if state.client_id in keep else state
            for state in live
        ]


def _load_run_metadata(runs_root: Path, run_id: str, role: str) -> RunMetadata:
    """Load a run's recorded ``RunMetadata`` (both runs are gated on it)."""
    metadata_path = runs_root / "runs" / run_id / "metadata.json"
    if not metadata_path.is_file():
        raise _InvalidIntervention(f"{role}_run_not_found: {run_id}")
    try:
        return RunMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise _InvalidIntervention(
            f"{role}_metadata_invalid: {run_id} ({exc})"
        ) from exc


def _run_config_from_metadata(metadata: RunMetadata, role: str) -> RunConfig:
    """Rebuild a run's ``RunConfig`` from its recorded ``RunMetadata``."""
    try:
        # metadata.config is RunConfig.model_dump(exclude={"run_id"})
        return RunConfig.model_validate({"run_id": metadata.run_id, **metadata.config})
    except (TypeError, ValueError, ValidationError) as exc:
        raise _InvalidIntervention(
            f"{role}_metadata_invalid: {metadata.run_id} ({exc})"
        ) from exc


def _check_compatible_runs(target: RunMetadata, source: RunMetadata) -> str | None:
    """Restore/inject gate: the two runs must be a matched pair (finding 4).

    Shape equality alone says nothing about compatible coordinate meaning —
    different ``(num_classes, num_features)`` layouts can flatten to the same
    length. Require the same seed, an equal dataset config, and a config
    delta limited to ``failure``; anything else is ``incompatible_runs``.
    """
    if target.seed != source.seed:
        return f"incompatible_runs: seed differs ({target.seed} != {source.seed})"
    target_config = dict(target.config)
    source_config = dict(source.config)
    if target_config.get("dataset") != source_config.get("dataset"):
        return "incompatible_runs: dataset config differs"
    target_config.pop("failure", None)
    source_config.pop("failure", None)
    if target_config != source_config:
        delta = sorted(
            key
            for key in set(target_config) | set(source_config)
            if target_config.get(key) != source_config.get(key)
        )
        return f"incompatible_runs: config delta beyond 'failure': {delta}"
    return None


def _make_recorder(runs_root: Path, run_id: str, role: str) -> Recorder:
    try:
        return Recorder(runs_root, run_id)
    except (ValueError, OSError) as exc:
        raise _InvalidIntervention(
            f"invalid_run_id: {role} run {run_id!r} ({exc})"
        ) from exc


def _load_recorded(recorder: Recorder, run_id: str, round_id: int, stage: str, role: str):
    try:
        return recorder.load(round_id, stage)
    except FileNotFoundError:
        raise _InvalidIntervention(
            f"{role}_boundary_missing: run {run_id!r} has no recorded state at "
            f"round {round_id} stage {stage!r}"
        ) from None
    except _LOAD_ERRORS as exc:
        # corrupt content hash, invalid/unknown model payload, broken npz, ...
        raise _InvalidIntervention(
            f"{role}_boundary_invalid: run {run_id!r} state at round {round_id} "
            f"stage {stage!r} failed to load ({exc})"
        ) from exc


def _validate_loaded_state(state, stage: str, role: str, run_id: str, round_id: int) -> None:
    """The recorded boundary must have the stage's type and round (finding 5).

    A file copied across stages can load as a valid model of the WRONG type
    (and crash the overlay with ``AttributeError``); a hand-crafted recording
    can claim a round outside the one it is stored under (and crash indexing
    later). Both are validation failures with stable reasons.
    """
    expected = _EXPECTED_STATE_TYPE[stage]
    if stage in _LIST_STAGES:
        entries = state if isinstance(state, list) else None
    else:
        entries = [state] if isinstance(state, expected) else None
    if entries is None or any(not isinstance(entry, expected) for entry in entries):
        suffix = " list" if stage in _LIST_STAGES else ""
        raise _InvalidIntervention(
            f"{role}_state_type_mismatch: run {run_id!r} round {round_id} "
            f"stage {stage!r} did not load as {expected.__name__}{suffix}"
        )
    rogue = [entry.round_id for entry in entries if entry.round_id != round_id]
    if rogue:
        raise _InvalidIntervention(
            f"{role}_round_mismatch: run {run_id!r} state stored under round "
            f"{round_id} stage {stage!r} claims round_id(s) {rogue}"
        )


def _validate_scope(spec: InterventionSpecification) -> list[str] | None:
    """Scope is absent or ``{"client_ids": [<unique nonempty strings>]}``."""
    unknown_scope = sorted(set(spec.scope) - {"client_ids"})
    if unknown_scope:
        raise _InvalidIntervention(
            f"invalid_scope: unsupported scope keys {unknown_scope}"
        )
    if "client_ids" not in spec.scope:
        return None
    if spec.stage not in _LIST_STAGES:
        raise _InvalidIntervention(
            f"invalid_scope: client_ids scope is only valid for "
            f"{list(_LIST_STAGES)}, got stage {spec.stage!r}"
        )
    raw = spec.scope["client_ids"]
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(cid, str) or not cid for cid in raw)
        or len(set(raw)) != len(raw)
    ):
        raise _InvalidIntervention(
            "invalid_scope: client_ids must be a nonempty list of unique "
            f"nonempty strings, got {raw!r}"
        )
    return list(raw)


def _round_trip_through_serialization(state, round_id: int, stage: str):
    """The same state after a Recorder save/load round-trip in a temp dir."""
    with tempfile.TemporaryDirectory(prefix="falcon_sham_") as tmp:
        recorder = Recorder(Path(tmp), "sham")
        recorder.record(round_id, stage, state)
        return recorder.load(round_id, stage)


def _run_translated(cfg: RunConfig, recorder=None, overlay=None):
    """Re-execute ``cfg``, translating infrastructure errors to invalid reasons."""
    try:
        return run(cfg, recorder=recorder, rng=Rng(cfg.seed), overlay=overlay)
    except _InvalidIntervention:
        raise
    except _LOAD_ERRORS as exc:
        raise _InvalidIntervention(f"run_reexecution_failed: {exc}") from exc


def _check_replay_fidelity(cfg: RunConfig, target_recorder: Recorder) -> str | None:
    """Sham gate (a): a NO-overlay replay must reproduce every recorded hash.

    Returns ``None`` when the unmodified replay of ``cfg`` matches the
    target's recording at every boundary; otherwise the drift reason
    ``replay_drift:<round>/<stage>`` for the first divergent boundary in
    pipeline order. A sham over a drifting replay certifies nothing.
    """
    with tempfile.TemporaryDirectory(prefix="falcon_replay_") as tmp:
        replay_recorder = Recorder(Path(tmp), "replay")
        _run_translated(cfg, recorder=replay_recorder)
        replay_hashes = replay_recorder.stage_hashes()
    try:
        target_hashes = target_recorder.stage_hashes()
    except _LOAD_ERRORS as exc:
        raise _InvalidIntervention(f"target_recording_invalid: {exc}") from exc
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    boundaries = sorted(
        set(replay_hashes) | set(target_hashes),
        key=lambda key: (key[0], stage_order[key[1]]),
    )
    for round_id, stage in boundaries:
        if replay_hashes.get((round_id, stage)) != target_hashes.get((round_id, stage)):
            return f"replay_drift:{round_id}/{stage}"
    return None


def _load_and_validate(
    recorder: Recorder,
    run_id: str,
    round_id: int,
    stage: str,
    role: str,
    tag_round: bool,
):
    """Load one recorded boundary and validate it; window mode tags the round.

    With ``tag_round`` (window mode, T13) any failure is re-raised with
    ``:<round>`` appended, so the reason names the window round whose
    recording rejected the whole spec.
    """
    try:
        state = _load_recorded(recorder, run_id, round_id, stage, role)
        _validate_loaded_state(state, stage, role, run_id, round_id)
    except _InvalidIntervention as exc:
        if tag_round:
            raise _InvalidIntervention(f"{exc}:{round_id}") from None
        raise
    return state


def apply_intervention(
    spec: InterventionSpecification, runs_root: Path
) -> InterventionResult:
    """Apply ``spec`` and re-execute the target run with the boundary replaced.

    With ``spec.round_window = [t1, t2]`` (T13, Plan §13.5) the replacement
    happens at EVERY round in the inclusive window (``round_id`` is ignored):
    one replay, per-round recorded source states, per-round validation as in
    the single-round case. Any invalid round rejects the whole intervention
    (``reason`` ends with ``:<round>``) — no partial windows.

    All validation failures return ``InterventionResult(valid=False,
    reason=...)`` — this function never raises for them.
    """
    runs_root = Path(runs_root)

    def invalid(reason: str) -> InterventionResult:
        return InterventionResult(spec=spec, valid=False, reason=reason)

    try:
        client_ids = _validate_scope(spec)

        target_metadata = _load_run_metadata(runs_root, spec.target_run_id, "target")
        cfg = _run_config_from_metadata(target_metadata, "target")
        window = spec.round_window
        if window is not None:
            t1, t2 = window
            if t1 < 0 or t1 > t2 or t2 >= cfg.rounds:
                raise _InvalidIntervention(
                    f"invalid_round_window: round_window {t1}:{t2} out of range "
                    f"for a run with {cfg.rounds} rounds"
                )
            rounds = list(range(t1, t2 + 1))
        else:
            if not 0 <= spec.round_id < cfg.rounds:
                raise _InvalidIntervention(
                    f"invalid_round_id: round_id {spec.round_id} out of range for "
                    f"a run with {cfg.rounds} rounds"
                )
            rounds = [spec.round_id]

        # the target must have recorded every intervention boundary, as the
        # stage's state type, and under the requested round(s)
        target_recorder = _make_recorder(runs_root, spec.target_run_id, "target")
        for round_id in rounds:
            _load_and_validate(
                target_recorder, spec.target_run_id, round_id, spec.stage,
                "target", window is not None,
            )

        if spec.mode in ("restore", "inject"):
            # identical machinery; the direction lives in which run is
            # target/source (Plan §13.1–13.2)
            source_metadata = _load_run_metadata(runs_root, spec.source_run_id, "source")
            incompatible = _check_compatible_runs(target_metadata, source_metadata)
            if incompatible is not None:
                raise _InvalidIntervention(incompatible)
            source_recorder = _make_recorder(runs_root, spec.source_run_id, "source")
            replacements = {
                round_id: _load_and_validate(
                    source_recorder, spec.source_run_id, round_id, spec.stage,
                    "source", window is not None,
                )
                for round_id in rounds
            }
            overlay = _ReplacementOverlay(
                replacements, spec.stage, client_ids, tag_round=window is not None
            )
        else:  # sham: source_run_id ignored content-wise (Plan §12.4)
            # (a) the unmodified replay must first prove itself drift-free
            drift = _check_replay_fidelity(cfg, target_recorder)
            if drift is not None:
                raise _InvalidIntervention(drift)
            # (b) then the LIVE boundary — never the recorded one — is
            # round-tripped through serialization and overlaid, at every
            # intervention round
            overlay = _ShamOverlay(rounds, spec.stage, client_ids)

        outcomes = _run_translated(cfg, overlay=overlay)
        if overlay.fired != len(rounds):
            raise _InvalidIntervention(
                f"overlay_misfire: overlay fired {overlay.fired} times at "
                f"round(s) {rounds} stage {spec.stage!r}, expected {len(rounds)}"
            )

        # final round's metrics plus "round_<t>_<metric>" per boundary round
        # (single round: the intervention round; window: t1 and t2)
        outcome_metrics: dict[str, float] = outcomes[-1].flat_metrics()
        boundary_rounds = rounds if window is None else sorted({rounds[0], rounds[-1]})
        for round_id in boundary_rounds:
            for key, value in outcomes[round_id].flat_metrics().items():
                outcome_metrics[f"round_{round_id}_{key}"] = value
        if spec.mode == "sham":
            # a sham must reproduce the unmodified target run; report the deviation
            if spec.stage == "evaluation":
                # (c) self-replacing the outcome is tautological: compare the
                # RECOMPUTED (pre-overlay) outcome against the recording at
                # the first intervention round instead
                recomputed = overlay.live_states[rounds[0]]
                recorded_at_round = _load_recorded(
                    target_recorder, spec.target_run_id, rounds[0], spec.stage, "target"
                )
                recorded_flat = recorded_at_round.flat_metrics()
                for key, value in recomputed.flat_metrics().items():
                    outcome_metrics[f"sham_deviation_{key}"] = (
                        value - recorded_flat.get(key, float("nan"))
                    )
            else:
                recorded_final = _load_recorded(
                    target_recorder, spec.target_run_id, cfg.rounds - 1, "evaluation", "target"
                )
                recorded_flat = recorded_final.flat_metrics()
                for key, value in outcomes[-1].flat_metrics().items():
                    outcome_metrics[f"sham_deviation_{key}"] = (
                        value - recorded_flat.get(key, float("nan"))
                    )

        return InterventionResult(spec=spec, valid=True, outcome_metrics=outcome_metrics)
    except _InvalidIntervention as exc:
        return invalid(str(exc))
