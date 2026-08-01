"""Pure attribution analysis over matched-pair schema objects."""

from falcon.metrics.effects import bis, failure_gap, nsie, nsre, sie, sre
from falcon.schema import (
    STAGES,
    AttributionReport,
    InterventionResult,
    PairValidationReport,
)


def _sham_deviation(
    result: InterventionResult,
    value: float,
    pair: PairValidationReport,
    m_ref: float,
    m_fail: float,
    higher_is_better: bool,
) -> float:
    baseline = m_ref if result.spec.target_run_id == pair.reference_run_id else m_fail
    direction = 1.0 if higher_is_better else -1.0
    return direction * (value - baseline)


def attribute(
    pair: PairValidationReport,
    interventions: list[InterventionResult],
    *,
    metric: str,
    m_ref: float,
    m_fail: float,
    higher_is_better: bool = True,
    min_gap: float,
    sham_tolerance: float,
    epsilon_tie: float = 1e-9,
    bystander_threshold: float = 0.1,
) -> AttributionReport:
    """Rank stage origins from restore, inject, and sham outcomes."""
    notes: list[str] = []
    grouped: dict[str, dict[str, InterventionResult]] = {}
    for result in interventions:
        stage, mode = result.spec.stage, result.spec.mode
        if not result.valid:
            notes.append(f"INVALID_INTERVENTION:{stage}:{mode}")
        elif metric in result.outcome_metrics:
            grouped.setdefault(stage, {})[mode] = result

    gap = failure_gap(m_ref, m_fail, higher_is_better)
    stage_effects: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}

    for stage in STAGES:
        results = grouped.get(stage)
        if not results:
            continue

        effects: dict[str, float] = {}
        restore = results.get("restore")
        inject = results.get("inject")
        sham = results.get("sham")

        normalized_restore = None
        if restore is not None:
            restored = restore.outcome_metrics[metric]
            effects["SRE"] = sre(restored, m_fail, higher_is_better)
            normalized_restore = nsre(
                restored, m_ref, m_fail, higher_is_better, min_gap
            )
            if normalized_restore is not None:
                effects["nSRE"] = normalized_restore

        normalized_inject = None
        if inject is not None:
            injected = inject.outcome_metrics[metric]
            effects["SIE"] = sie(m_ref, injected, higher_is_better)
            normalized_inject = nsie(
                m_ref, injected, m_fail, higher_is_better, min_gap
            )
            if normalized_inject is not None:
                effects["nSIE"] = normalized_inject

        combined = bis(normalized_restore, normalized_inject)
        if combined is not None:
            effects["BIS"] = combined
            scores[stage] = combined
        elif normalized_restore is not None:
            scores[stage] = normalized_restore
        elif normalized_inject is not None:
            scores[stage] = normalized_inject

        if sham is not None:
            effects["sham_dev"] = _sham_deviation(
                sham,
                sham.outcome_metrics[metric],
                pair,
                m_ref,
                m_fail,
                higher_is_better,
            )

        stage_effects[stage] = effects

    report_args = {
        "pair": pair,
        "failure_gap": {metric: gap},
        "stage_effects": stage_effects,
        "notes": notes,
    }

    if abs(gap) < min_gap:
        notes.append("INSUFFICIENT_FAILURE_GAP")
        return AttributionReport(origin_ranking=[], roles={}, **report_args)

    sham_violations = [
        stage
        for stage in STAGES
        if abs(stage_effects.get(stage, {}).get("sham_dev", 0.0))
        > sham_tolerance
    ]
    if sham_violations:
        notes.extend(f"SHAM_VIOLATION:{stage}" for stage in sham_violations)
        return AttributionReport(origin_ranking=[], roles={}, **report_args)

    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    ranking = sorted(scores, key=lambda stage: (-scores[stage], stage_order[stage]))
    max_score = scores[ranking[0]] if ranking else None
    if ranking:
        tied = [
            stage
            for stage in ranking
            if abs(scores[ranking[0]] - scores[stage]) <= epsilon_tie
        ]
        ranking[: len(tied)] = sorted(tied, key=stage_order.__getitem__)

    first = pair.first_divergence_stage
    if first in scores and max_score is not None and scores[first] >= max_score - epsilon_tie:
        ranking.remove(first)
        ranking.insert(0, first)

    if len(ranking) >= 2 and abs(scores[ranking[0]] - scores[ranking[1]]) <= epsilon_tie:
        notes.append(f"UNRESOLVED_BETWEEN:{ranking[0]},{ranking[1]}")

    roles: dict[str, str] = {}
    for index, stage in enumerate(ranking):
        score = scores[stage]
        if index == 0:
            roles[stage] = "origin_candidate"
        elif score < 0:
            roles[stage] = "suppressor_candidate"
        elif score < bystander_threshold:
            roles[stage] = "bystander"
        else:
            roles[stage] = "carrier_or_amplifier"

    return AttributionReport(origin_ranking=ranking, roles=roles, **report_args)
