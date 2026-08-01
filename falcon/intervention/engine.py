"""Intervention engine: restore / inject / sham (Task T5, Plan §11.5, §13.1–13.3).

Replay strategy (PM decision): full deterministic re-execution with an
overlay — re-run ``run(cfg)`` from round 0 and, at the intervention boundary,
swap the produced stage state for the recorded source state, then continue
downstream. No checkpoint restoration.

``runs_root`` is the recorder root directory: recorded runs live at
``runs_root/runs/<run_id>/`` (``metadata.json`` + per-round stage files), the
same layout ``falcon.recorder.recorder.Recorder(root_dir, run_id)`` writes.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    InterventionResult,
    InterventionSpecification,
    RunConfig,
    RunMetadata,
)

#: per-client stages recorded once per stage as a list (CONTRACTS §1)
_LIST_STAGES = ("local", "compression")
#: lineage hash field per list stage, checked against the live replay
_LINEAGE_FIELD = {"local": "base_model_hash", "compression": "uncompressed_hash"}
#: array field per stage, shape-checked against the live computed state
_ARRAY_FIELD = {"local": "update", "compression": "update", "aggregation": "aggregate"}

#: outcome_metrics key set (to 1.0) when the replaced entries' lineage hashes
#: do not match the live replay's — a WARNING, never fatal (Plan §13)
WARNING_BASE_MODEL_MISMATCH = "warning_base_model_mismatch"


class _InvalidIntervention(Exception):
    """Internal abort signal: a validation check failed at the live boundary.

    Raised by the overlay during re-execution and caught by
    ``apply_intervention`` — validation failures must surface as
    ``InterventionResult(valid=False, reason=...)``, never as exceptions.
    """


class _ReplacementOverlay:
    """Runner overlay swapping one boundary state for a recorded replacement.

    Validation that needs the live computed state happens here, at the
    boundary: array shapes (vs the live state), client identity, scoped
    client_ids presence in BOTH live and source, and the lineage-hash check
    (recorded as a warning flag, replacement still proceeds).
    """

    def __init__(
        self,
        round_id: int,
        stage: str,
        replacement,
        client_ids: list[str] | None,
    ):
        self._round_id = round_id
        self._stage = stage
        self._replacement = replacement
        self._client_ids = client_ids  # None => whole-stage replacement
        self.base_model_mismatch = False

    def override(self, round_id: int, stage: str, state):
        if round_id != self._round_id or stage != self._stage:
            return state
        if stage in _LIST_STAGES:
            return self._override_list(state)
        return self._override_single(state)

    # --- single states: selection / aggregation / evaluation ---

    def _override_single(self, live):
        array_field = _ARRAY_FIELD.get(self._stage)
        if array_field is not None:
            self._check_shape(
                getattr(self._replacement, array_field),
                getattr(live, array_field),
                context=f"stage {self._stage!r}",
            )
        return self._replacement

    # --- list states: local / compression ---

    def _override_list(self, live: list) -> list:
        source = self._replacement
        live_by_id = {state.client_id: state for state in live}
        source_by_id = {state.client_id: state for state in source}

        if self._client_ids is None:
            live_ids = [state.client_id for state in live]
            source_ids = [state.client_id for state in source]
            if source_ids != live_ids:
                raise _InvalidIntervention(
                    f"client_mismatch: source client_ids {source_ids} do not "
                    f"match live client_ids {live_ids} at round "
                    f"{self._round_id} stage {self._stage!r}"
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
                    f"in BOTH live and source state at round {self._round_id} "
                    f"stage {self._stage!r}"
                )
            replaced_ids = self._client_ids

        for cid in replaced_ids:
            live_entry = live_by_id[cid]
            source_entry = source_by_id[cid]
            self._check_shape(
                getattr(source_entry, _ARRAY_FIELD[self._stage]),
                getattr(live_entry, _ARRAY_FIELD[self._stage]),
                context=f"stage {self._stage!r} client {cid!r}",
            )
            lineage_field = _LINEAGE_FIELD[self._stage]
            if getattr(source_entry, lineage_field) != getattr(live_entry, lineage_field):
                # expected when replacing cross-run states (Plan §13): surface
                # as a warning in outcome_metrics, but proceed.
                self.base_model_mismatch = True

        if self._client_ids is None:
            return list(source)
        replaced = set(self._client_ids)
        return [
            source_by_id[state.client_id] if state.client_id in replaced else state
            for state in live
        ]

    def _check_shape(self, source_array, live_array, context: str) -> None:
        source_shape = tuple(np.shape(source_array))
        live_shape = tuple(np.shape(live_array))
        if source_shape != live_shape:
            raise _InvalidIntervention(
                f"shape_mismatch: source shape {source_shape} != live shape "
                f"{live_shape} at round {self._round_id} {context}"
            )


def _load_target_config(runs_root: Path, run_id: str) -> RunConfig:
    """Rebuild a run's ``RunConfig`` from its recorded ``RunMetadata``."""
    metadata_path = runs_root / "runs" / run_id / "metadata.json"
    if not metadata_path.is_file():
        raise _InvalidIntervention(f"target_run_not_found: {run_id}")
    try:
        metadata = RunMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        # metadata.config is RunConfig.model_dump(exclude={"run_id"})
        return RunConfig.model_validate({"run_id": metadata.run_id, **metadata.config})
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        raise _InvalidIntervention(f"target_metadata_invalid: {run_id} ({exc})") from exc


def _load_recorded(recorder: Recorder, run_id: str, round_id: int, stage: str, role: str):
    try:
        return recorder.load(round_id, stage)
    except FileNotFoundError:
        raise _InvalidIntervention(
            f"{role}_boundary_missing: run {run_id!r} has no recorded state at "
            f"round {round_id} stage {stage!r}"
        ) from None


def _round_trip_through_serialization(state, round_id: int, stage: str):
    """Sham replacement: the same state after a Recorder save/load round-trip."""
    with tempfile.TemporaryDirectory(prefix="falcon_sham_") as tmp:
        recorder = Recorder(Path(tmp), "sham")
        recorder.record(round_id, stage, state)
        return recorder.load(round_id, stage)


def apply_intervention(
    spec: InterventionSpecification, runs_root: Path
) -> InterventionResult:
    """Apply ``spec`` and re-execute the target run with the boundary replaced.

    All validation failures return ``InterventionResult(valid=False,
    reason=...)`` — this function never raises for them.
    """
    runs_root = Path(runs_root)

    def invalid(reason: str) -> InterventionResult:
        return InterventionResult(spec=spec, valid=False, reason=reason)

    try:
        # scope: {} = whole stage; {"client_ids": [...]} for list stages only
        unknown_scope = sorted(set(spec.scope) - {"client_ids"})
        if unknown_scope:
            return invalid(f"invalid_scope: unsupported scope keys {unknown_scope}")
        client_ids = None
        if "client_ids" in spec.scope:
            if spec.stage not in _LIST_STAGES:
                return invalid(
                    f"invalid_scope: client_ids scope is only valid for "
                    f"{list(_LIST_STAGES)}, got stage {spec.stage!r}"
                )
            client_ids = [str(cid) for cid in spec.scope["client_ids"]]

        cfg = _load_target_config(runs_root, spec.target_run_id)

        # the target must have recorded the intervention boundary (it is also
        # the sham replacement's content source)
        target_recorder = Recorder(runs_root, spec.target_run_id)
        target_state = _load_recorded(
            target_recorder, spec.target_run_id, spec.round_id, spec.stage, "target"
        )

        if spec.mode in ("restore", "inject"):
            # identical machinery; the direction lives in which run is
            # target/source (Plan §13.1–13.2)
            if not (runs_root / "runs" / spec.source_run_id).is_dir():
                return invalid(f"source_run_not_found: {spec.source_run_id}")
            source_recorder = Recorder(runs_root, spec.source_run_id)
            replacement = _load_recorded(
                source_recorder, spec.source_run_id, spec.round_id, spec.stage, "source"
            )
        else:  # sham: source_run_id ignored content-wise (Plan §12.4)
            replacement = _round_trip_through_serialization(
                target_state, spec.round_id, spec.stage
            )

        overlay = _ReplacementOverlay(spec.round_id, spec.stage, replacement, client_ids)
        outcomes = run(cfg, rng=Rng(cfg.seed), overlay=overlay)
    except _InvalidIntervention as exc:
        return invalid(str(exc))

    # final round's metrics plus "round_<t>_<metric>" for the intervention round
    outcome_metrics: dict[str, float] = dict(outcomes[-1].metrics)
    for key, value in outcomes[spec.round_id].metrics.items():
        outcome_metrics[f"round_{spec.round_id}_{key}"] = value
    if overlay.base_model_mismatch:
        outcome_metrics[WARNING_BASE_MODEL_MISMATCH] = 1.0
    if spec.mode == "sham":
        # a sham must reproduce the unmodified target run; report the deviation
        recorded_final = target_recorder.load(cfg.rounds - 1, "evaluation")
        for key, value in outcomes[-1].metrics.items():
            outcome_metrics[f"sham_deviation_{key}"] = value - recorded_final.metrics.get(
                key, float("nan")
            )

    return InterventionResult(spec=spec, valid=True, outcome_metrics=outcome_metrics)
