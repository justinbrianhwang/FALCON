"""End-to-end orchestration for matched-run attribution."""

from __future__ import annotations

import json
from pathlib import Path

from falcon.attribution.analyzer import attribute
from falcon.intervention import apply_intervention
from falcon.matcher.matcher import validate_pair
from falcon.recorder import Recorder
from falcon.schema import (
    AttributionReport,
    FailureSpecification,
    InterventionResult,
    InterventionSpecification,
    RunMetadata,
)

_INTERVENABLE_STAGES = ("selection", "local", "compression", "aggregation")


def _metadata(runs_root: Path, run_id: str) -> RunMetadata:
    path = runs_root / "runs" / run_id / "metadata.json"
    return RunMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_ground_truth(runs_root: Path, failure_run_id: str) -> FailureSpecification | None:
    """Load the benchmark failure specification recorded for a run."""
    return _metadata(Path(runs_root), failure_run_id).failure


def analyze_pair(
    runs_root: Path,
    reference_run_id: str,
    failure_run_id: str,
    *,
    metric: str,
    higher_is_better: bool,
    min_gap: float,
    sham_tolerance: float,
    rounds: list[int] | None = None,
) -> tuple[AttributionReport, list[InterventionResult]]:
    """Validate, intervene on, and attribute a recorded matched run pair."""
    runs_root = Path(runs_root)
    pair = validate_pair(
        runs_root / "runs" / reference_run_id,
        runs_root / "runs" / failure_run_id,
    )
    if pair.status == "INVALID_PAIR":
        return (
            AttributionReport(
                pair=pair,
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                roles={},
                notes=["INVALID_PAIR"],
            ),
            [],
        )

    reference = _metadata(runs_root, reference_run_id)
    failure = _metadata(runs_root, failure_run_id)
    if rounds is None:
        chosen_round = pair.first_divergence_round
        if chosen_round is None:
            if failure.failure is None:
                raise ValueError("matched failure run has no failure specification")
            chosen_round = failure.failure.active_rounds[0]
        rounds = [chosen_round]

    interventions = [
        apply_intervention(
            InterventionSpecification(
                target_run_id=(reference_run_id if mode == "inject" else failure_run_id),
                source_run_id=(failure_run_id if mode == "inject" else reference_run_id),
                round_id=round_id,
                stage=stage,
                mode=mode,
            ),
            runs_root,
        )
        for round_id in rounds
        for stage in _INTERVENABLE_STAGES
        for mode in ("restore", "inject", "sham")
    ]

    reference_outcome = Recorder(runs_root, reference_run_id).load(
        reference.rounds - 1, "evaluation"
    )
    failure_outcome = Recorder(runs_root, failure_run_id).load(
        failure.rounds - 1, "evaluation"
    )
    report = attribute(
        pair,
        interventions,
        metric=metric,
        m_ref=reference_outcome.metrics[metric],
        m_fail=failure_outcome.metrics[metric],
        higher_is_better=higher_is_better,
        min_gap=min_gap,
        sham_tolerance=sham_tolerance,
    )
    return report, interventions
