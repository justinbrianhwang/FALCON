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


def load_ground_truth(
    runs_root: Path, failure_run_id: str
) -> FailureSpecification | list[FailureSpecification] | None:
    """Load the benchmark failure specification recorded for a run."""
    metadata = _metadata(Path(runs_root), failure_run_id)
    return metadata.failure or metadata.failures or None


def _persistent_divergence_window(
    runs_root: Path, reference_run_id: str, failure_run_id: str
) -> tuple[int, int] | None:
    reference = Recorder(runs_root, reference_run_id).stage_hashes()
    failure = Recorder(runs_root, failure_run_id).stage_hashes()
    divergent_rounds = sorted(
        {
            round_id
            for round_id, stage in reference.keys() | failure.keys()
            if reference.get((round_id, stage)) != failure.get((round_id, stage))
        }
    )
    if len(divergent_rounds) > 1:
        return divergent_rounds[0], divergent_rounds[-1]
    return None


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
    epsilon_tie: float = 1e-9,
    decisive_margin: float = 0.05,
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
                outcome="invalid_pair",
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                origin_set=[],
                roles={},
                notes=["INVALID_PAIR"],
            ),
            [],
        )

    try:
        reference = _metadata(runs_root, reference_run_id)
        failure = _metadata(runs_root, failure_run_id)
    except Exception as exc:
        return (
            AttributionReport(
                pair=pair,
                outcome="unresolved",
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                origin_set=[],
                roles={},
                notes=[f"INVALID_METADATA:{type(exc).__name__}"],
            ),
            [],
        )
    round_window: tuple[int, int] | None = None
    if rounds is None:
        chosen_round = pair.first_divergence_round
        if failure.failure is None and failure.failures:
            round_window = (
                min(spec.active_rounds[0] for spec in failure.failures),
                max(spec.active_rounds[1] for spec in failure.failures),
            )
            if chosen_round is None:
                chosen_round = round_window[0]
        else:
            if chosen_round is None:
                if failure.failure is None:
                    return (
                        AttributionReport(
                            pair=pair,
                            outcome="unresolved",
                            failure_gap={},
                            stage_effects={},
                            origin_ranking=[],
                            origin_set=[],
                            roles={},
                            notes=["MISSING_FAILURE_SPECIFICATION"],
                        ),
                        [],
                    )
                chosen_round = failure.failure.active_rounds[0]
            failure_window = failure.failure.active_rounds if failure.failure else None
            if failure_window and failure_window[0] < failure_window[1]:
                round_window = failure_window
            else:
                round_window = _persistent_divergence_window(
                    runs_root, reference_run_id, failure_run_id
                )
        if chosen_round is None:
            return (
                AttributionReport(
                    pair=pair,
                    outcome="unresolved",
                    failure_gap={},
                    stage_effects={},
                    origin_ranking=[],
                    origin_set=[],
                    roles={},
                    notes=["MISSING_FAILURE_SPECIFICATION"],
                ),
                [],
            )
        rounds = [round_window[0] if round_window else chosen_round]

    try:
        reference_outcome = Recorder(runs_root, reference_run_id).load(
            reference.rounds - 1, "evaluation"
        )
        failure_outcome = Recorder(runs_root, failure_run_id).load(
            failure.rounds - 1, "evaluation"
        )
        m_ref = reference_outcome.flat_metrics()[metric]
        m_fail = failure_outcome.flat_metrics()[metric]
    except KeyError:
        return (
            AttributionReport(
                pair=pair,
                outcome="unresolved",
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                origin_set=[],
                roles={},
                notes=[f"MISSING_METRIC:{metric}"],
            ),
            [],
        )
    except Exception as exc:
        return (
            AttributionReport(
                pair=pair,
                outcome="unresolved",
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                origin_set=[],
                roles={},
                notes=[f"INVALID_EVALUATION:{type(exc).__name__}"],
            ),
            [],
        )

    interventions: list[InterventionResult] = []
    for round_id in rounds:
        for stage in _INTERVENABLE_STAGES:
            for mode in ("restore", "inject", "sham"):
                spec = InterventionSpecification(
                    target_run_id=(
                        reference_run_id if mode == "inject" else failure_run_id
                    ),
                    source_run_id=(
                        failure_run_id if mode == "inject" else reference_run_id
                    ),
                    round_id=round_id,
                    round_window=round_window,
                    stage=stage,
                    mode=mode,
                )
                try:
                    result = apply_intervention(spec, runs_root)
                except Exception as exc:
                    result = InterventionResult(
                        spec=spec,
                        valid=False,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                interventions.append(result)

    report = attribute(
        pair,
        interventions,
        metric=metric,
        m_ref=m_ref,
        m_fail=m_fail,
        higher_is_better=higher_is_better,
        min_gap=min_gap,
        sham_tolerance=sham_tolerance,
        epsilon_tie=epsilon_tie,
        decisive_margin=decisive_margin,
    )
    if failure.failures and report.outcome == "unique_origin":
        first_origin_with_carriers = (
            bool(report.origin_ranking)
            and report.origin_ranking[0] == pair.first_divergence_stage
            and any(
                note.startswith("CARRIER_TIE_RESOLVED:") for note in report.notes
            )
        )
        if not first_origin_with_carriers:
            report.outcome = "unresolved"
            report.origin_set = list(report.origin_ranking)
            report.roles.update(
                {stage: "unresolved_candidate" for stage in report.origin_set}
            )
            report.notes.append("COMPOUND_FAILURE_AMBIGUITY")
    return report, interventions


def analyze_pair_vector(
    runs_root: Path,
    reference_run_id: str,
    failure_run_id: str,
    *,
    metrics: dict[str, dict],
    min_gap: float,
    sham_tolerance: float,
    rounds: list[int] | None = None,
    epsilon_tie: float = 1e-9,
    decisive_margin: float = 0.05,
) -> tuple[dict[str, AttributionReport], list[InterventionResult]]:
    """Metric-specific attribution over ONE shared intervention set (Plan 14.10).

    ``metrics`` maps metric name to ``{"higher_is_better": bool}`` with an
    optional per-metric ``"min_gap"`` overriding the shared default. The first
    metric drives the (expensive) intervention replay; every other metric is a
    pure re-analysis of the same evidence, so the vector costs no more replay
    than a single-metric run. Put a metric that is always recorded (e.g.
    ``accuracy``) first — an early-exit on the first metric returns no
    interventions to re-analyze.
    """
    runs_root = Path(runs_root)
    names = list(metrics)
    first = names[0]
    report, interventions = analyze_pair(
        runs_root,
        reference_run_id,
        failure_run_id,
        metric=first,
        higher_is_better=metrics[first]["higher_is_better"],
        min_gap=metrics[first].get("min_gap", min_gap),
        sham_tolerance=sham_tolerance,
        rounds=rounds,
        epsilon_tie=epsilon_tie,
        decisive_margin=decisive_margin,
    )
    reports = {first: report}
    if not interventions:
        return reports, interventions

    reference = _metadata(runs_root, reference_run_id)
    failure = _metadata(runs_root, failure_run_id)
    ref_flat = Recorder(runs_root, reference_run_id).load(
        reference.rounds - 1, "evaluation"
    ).flat_metrics()
    fail_flat = Recorder(runs_root, failure_run_id).load(
        failure.rounds - 1, "evaluation"
    ).flat_metrics()
    for name in names[1:]:
        if name not in ref_flat or name not in fail_flat:
            reports[name] = AttributionReport(
                pair=report.pair,
                outcome="unresolved",
                failure_gap={},
                stage_effects={},
                origin_ranking=[],
                origin_set=[],
                roles={},
                notes=[f"MISSING_METRIC:{name}"],
            )
            continue
        reports[name] = attribute(
            report.pair,
            interventions,
            metric=name,
            m_ref=ref_flat[name],
            m_fail=fail_flat[name],
            higher_is_better=metrics[name]["higher_is_better"],
            min_gap=metrics[name].get("min_gap", min_gap),
            sham_tolerance=sham_tolerance,
            epsilon_tie=epsilon_tie,
            decisive_margin=decisive_margin,
        )
    return reports, interventions
