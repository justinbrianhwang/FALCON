"""Validation of matched reference and failure run directories."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from falcon.recorder import Recorder, hash_model
from falcon.schema import (
    STAGES,
    FailureSpecification,
    PairValidationReport,
    RunMetadata,
)


def _sidecar(base: Path, suffix: str) -> Path:
    return base.parent / f"{base.name}{suffix}"


def _load_metadata(run_dir: Path) -> RunMetadata | None:
    try:
        return RunMetadata.model_validate_json(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, ValidationError):
        return None


def _boundaries(run_dir: Path) -> tuple[set[tuple[int, str]], bool]:
    result: set[tuple[int, str]] = set()
    layout_valid = True
    for round_dir in run_dir.glob("round_*"):
        if not round_dir.is_dir():
            continue
        try:
            round_id = int(round_dir.name.removeprefix("round_"))
        except ValueError:
            continue
        for stage in STAGES:
            base = round_dir / stage
            json_path = _sidecar(base, ".json")
            npz_path = _sidecar(base, ".npz")
            if base.is_dir():
                result.add((round_id, stage))
                json_stems = {path.stem for path in base.glob("*.json")}
                npz_stems = {path.stem for path in base.glob("*.npz")}
                layout_valid &= not (json_path.exists() or npz_path.exists())
                layout_valid &= npz_stems <= json_stems
            elif json_path.is_file():
                result.add((round_id, stage))
            elif npz_path.exists():
                layout_valid = False
    return result, layout_valid


def _stage_hashes(
    run_dir: Path, boundaries: set[tuple[int, str]]
) -> dict[tuple[int, str], str]:
    recorder = Recorder.__new__(Recorder)
    recorder.run_id = run_dir.name
    recorder.run_dir = run_dir

    hashes: dict[tuple[int, str], str] = {}
    for boundary in boundaries:
        state = recorder.load(*boundary)
        if isinstance(state, list):
            payload = [hash_model(model) for model in state]
            hashes[boundary] = sha256(
                json.dumps(payload, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        else:
            hashes[boundary] = hash_model(state)
    return hashes


def _failure_only_delta(
    reference: RunMetadata, failure: RunMetadata
) -> bool:
    reference_config = dict(reference.config)
    failure_config = dict(failure.config)
    if failure.failure is not None and not failure.failures:
        key = "failure"
        configured = failure.failure
    elif failure.failure is None and failure.failures:
        key = "failures"
        configured = failure.failures
    else:
        return False

    if key not in failure_config:
        return False
    reference_failure = reference_config.pop(key, None)
    configured_failure = failure_config.pop(key)
    try:
        if key == "failure":
            parsed_failure = FailureSpecification.model_validate(configured_failure)
        else:
            parsed_failure = [
                FailureSpecification.model_validate(spec)
                for spec in configured_failure
            ]
    except (TypeError, ValueError, ValidationError):
        return False

    return (
        reference.failure is None
        and not reference.failures
        and reference_failure in (None, [])
        and parsed_failure == configured
        and reference_config == failure_config
    )


def _selected_ids(run_dir: Path, round_id: int) -> list[str]:
    data = json.loads(
        (run_dir / f"round_{round_id}" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    selected_ids = data["selected_ids"]
    if not isinstance(selected_ids, list) or not all(
        isinstance(client_id, str) for client_id in selected_ids
    ):
        raise ValueError("invalid selected_ids")
    return selected_ids


def validate_pair(
    reference_dir: Path, failure_dir: Path
) -> PairValidationReport:
    """Validate that two recorded runs differ only by a controlled failure."""
    reference_dir = Path(reference_dir)
    failure_dir = Path(failure_dir)
    reference = _load_metadata(reference_dir)
    failure = _load_metadata(failure_dir)
    metadata_loadable = reference is not None and failure is not None

    checks = {
        "metadata_loadable": metadata_loadable,
        "same_seed": False,
        "same_rounds": False,
        "same_dataset_config": False,
        "same_code_version": False,
        "config_delta_is_failure_only": False,
        "stage_hash_coverage": False,
        "pre_failure_hashes_match": False,
        "selection_matches_pre_failure": False,
    }

    if metadata_loadable:
        assert reference is not None and failure is not None
        checks["same_seed"] = reference.seed == failure.seed
        checks["same_rounds"] = reference.rounds == failure.rounds
        checks["same_code_version"] = (
            reference.code_version == failure.code_version
        )
        checks["same_dataset_config"] = (
            "dataset" in reference.config
            and "dataset" in failure.config
            and reference.config["dataset"] == failure.config["dataset"]
        )
        checks["config_delta_is_failure_only"] = _failure_only_delta(
            reference, failure
        )

    reference_boundaries, reference_layout_valid = _boundaries(reference_dir)
    failure_boundaries, failure_layout_valid = _boundaries(failure_dir)
    checks["stage_hash_coverage"] = (
        reference_layout_valid
        and failure_layout_valid
        and reference_boundaries == failure_boundaries
    )

    reference_hashes: dict[tuple[int, str], str] = {}
    failure_hashes: dict[tuple[int, str], str] = {}
    hashes_loadable = True
    try:
        reference_hashes = _stage_hashes(reference_dir, reference_boundaries)
        failure_hashes = _stage_hashes(failure_dir, failure_boundaries)
    except Exception:
        hashes_loadable = False

    failure_specs = (
        ([failure.failure] if failure.failure is not None else failure.failures)
        if failure is not None
        else []
    )
    pre_failure: set[tuple[int, str]] = set()
    if hashes_loadable and failure_specs:
        start_round = min(spec.active_rounds[0] for spec in failure_specs)
        pre_failure = {
            boundary
            for boundary in reference_boundaries | failure_boundaries
            if boundary[0] < start_round
        }
        checks["pre_failure_hashes_match"] = all(
            reference_hashes.get(boundary) == failure_hashes.get(boundary)
            for boundary in pre_failure
        )
        try:
            checks["selection_matches_pre_failure"] = all(
                _selected_ids(reference_dir, round_id)
                == _selected_ids(failure_dir, round_id)
                for round_id in range(min(start_round, failure.rounds))
            )
        except (OSError, ValueError, KeyError, TypeError):
            checks["selection_matches_pre_failure"] = False

    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    divergent = []
    if hashes_loadable:
        divergent = [
            boundary
            for boundary in reference_boundaries | failure_boundaries
            if reference_hashes.get(boundary) != failure_hashes.get(boundary)
        ]
        divergent.sort(key=lambda item: (item[0], stage_order[item[1]]))
    first_round, first_stage = divergent[0] if divergent else (None, None)

    warnings: list[str] = []
    if metadata_loadable and not checks["same_code_version"]:
        warnings.append("code versions differ - causal comparison may be unreliable")
    if hashes_loadable and failure_specs and not pre_failure:
        warnings.append(
            "no pre-failure boundaries recorded - match rests on config/seed only"
        )
    if hashes_loadable and checks["stage_hash_coverage"] and not divergent:
        warnings.append("runs are identical - no failure effect recorded")
    if (
        first_stage is not None
        and failure_specs
        and first_stage not in {spec.stage for spec in failure_specs}
    ):
        configured_stages = [spec.stage for spec in failure_specs]
        warnings.append(
            f"first divergence stage {first_stage!r} differs from configured "
            f"failure stage(s) {configured_stages!r} in round {first_round}"
        )
    if (
        first_round is not None
        and failure_specs
        and not (
            min(spec.active_rounds[0] for spec in failure_specs)
            <= first_round
            <= max(spec.active_rounds[1] for spec in failure_specs)
        )
    ):
        failure_window = (
            min(spec.active_rounds[0] for spec in failure_specs),
            max(spec.active_rounds[1] for spec in failure_specs),
        )
        warnings.append(
            f"first divergence round {first_round} falls outside configured "
            f"failure window {failure_window}"
        )

    fatal_checks = {
        name: passed
        for name, passed in checks.items()
        if name != "same_code_version"
    }
    if not all(fatal_checks.values()):
        status = "INVALID_PAIR"
    elif warnings:
        status = "MATCHED_WITH_WARNINGS"
    else:
        status = "MATCHED"

    return PairValidationReport(
        reference_run_id=(
            reference.run_id if reference is not None else reference_dir.name
        ),
        failure_run_id=(
            failure.run_id if failure is not None else failure_dir.name
        ),
        status=status,
        checks=checks,
        warnings=warnings,
        first_divergence_round=first_round,
        first_divergence_stage=first_stage,
    )
