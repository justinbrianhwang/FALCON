"""Validation of matched reference and failure run directories."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

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


def _boundaries(run_dir: Path) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    for round_dir in run_dir.glob("round_*"):
        if not round_dir.is_dir():
            continue
        try:
            round_id = int(round_dir.name.removeprefix("round_"))
        except ValueError:
            continue
        for stage in STAGES:
            base = round_dir / stage
            if _sidecar(base, ".json").is_file() or base.is_dir():
                result.add((round_id, stage))
    return result


def _model_fingerprint(base: Path) -> tuple[int, str]:
    data = json.loads(_sidecar(base, ".json").read_text(encoding="utf-8"))
    index = data.get("__index__", 0)

    digest = sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    npz_path = _sidecar(base, ".npz")
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            for key in sorted(archive.files):
                array = archive[key]
                digest.update(key.encode("utf-8"))
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(str(array.shape).encode("ascii"))
                digest.update(array.tobytes(order="C"))
    return index, digest.hexdigest()


def _stage_fingerprint(run_dir: Path, boundary: tuple[int, str]) -> str:
    round_id, stage = boundary
    base = run_dir / f"round_{round_id}" / stage
    if base.is_dir():
        fingerprints = sorted(
            _model_fingerprint(path.with_suffix(""))
            for path in base.glob("*.json")
        )
        payload: Any = fingerprints
    else:
        payload = _model_fingerprint(base)
    return sha256(
        json.dumps(payload, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _stage_hashes(
    run_dir: Path, boundaries: set[tuple[int, str]]
) -> dict[tuple[int, str], str]:
    return {
        boundary: _stage_fingerprint(run_dir, boundary)
        for boundary in boundaries
    }


def _failure_only_delta(
    reference: RunMetadata, failure: RunMetadata
) -> bool:
    reference_config = dict(reference.config)
    failure_config = dict(failure.config)
    if "failure" not in failure_config:
        return False

    reference_failure = reference_config.pop("failure", None)
    configured_failure = failure_config.pop("failure")
    try:
        parsed_failure = FailureSpecification.model_validate(configured_failure)
    except (TypeError, ValueError, ValidationError):
        return False

    return (
        reference.failure is None
        and failure.failure is not None
        and reference_failure is None
        and parsed_failure == failure.failure
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
        "config_delta_is_failure_only": False,
        "stage_hash_coverage": False,
        "pre_failure_hashes_match": False,
        "selection_matches_pre_failure": False,
    }

    if metadata_loadable:
        assert reference is not None and failure is not None
        checks["same_seed"] = reference.seed == failure.seed
        checks["same_rounds"] = reference.rounds == failure.rounds
        checks["same_dataset_config"] = (
            "dataset" in reference.config
            and "dataset" in failure.config
            and reference.config["dataset"] == failure.config["dataset"]
        )
        checks["config_delta_is_failure_only"] = _failure_only_delta(
            reference, failure
        )

    reference_boundaries = _boundaries(reference_dir)
    failure_boundaries = _boundaries(failure_dir)
    checks["stage_hash_coverage"] = reference_boundaries == failure_boundaries

    reference_hashes: dict[tuple[int, str], str] = {}
    failure_hashes: dict[tuple[int, str], str] = {}
    hashes_loadable = True
    try:
        reference_hashes = _stage_hashes(reference_dir, reference_boundaries)
        failure_hashes = _stage_hashes(failure_dir, failure_boundaries)
    except (OSError, ValueError, KeyError, TypeError):
        hashes_loadable = False

    failure_spec = failure.failure if failure is not None else None
    if hashes_loadable and failure_spec is not None:
        start_round = failure_spec.active_rounds[0]
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
    if hashes_loadable and checks["stage_hash_coverage"] and not divergent:
        warnings.append("runs are identical - no failure effect recorded")
    if (
        first_stage is not None
        and failure_spec is not None
        and first_stage != failure_spec.stage
    ):
        warnings.append(
            f"first divergence stage {first_stage!r} differs from configured "
            f"failure stage {failure_spec.stage!r} in round {first_round}"
        )

    if not all(checks.values()):
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
