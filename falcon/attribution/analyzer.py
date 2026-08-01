"""Pure attribution analysis over matched-pair schema objects."""

from math import isfinite
from statistics import fmean

from falcon.metrics.effects import (
    bis,
    failure_gap,
    nsie,
    nsre,
    sham_adjusted,
    sie,
    sre,
)
from falcon.schema import (
    STAGES,
    AttributionReport,
    InterventionResult,
    PairValidationReport,
)


def _sham_deviation(
    value: float,
    m_fail: float,
    higher_is_better: bool,
) -> float:
    direction = 1.0 if higher_is_better else -1.0
    return direction * (value - m_fail)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


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
    decisive_margin: float = 0.05,
    bystander_threshold: float = 0.1,
) -> AttributionReport:
    """Rank stages, preferring bidirectional evidence when scores are equal.

    Propagation roles remain provisional pending the standardized deviations in
    Plan 14.9.
    """
    gap = failure_gap(m_ref, m_fail, higher_is_better)
    if pair.status == "INVALID_PAIR":
        return AttributionReport(
            pair=pair,
            outcome="invalid_pair",
            failure_gap={metric: gap},
            stage_effects={},
            origin_ranking=[],
            origin_set=[],
            roles={},
            notes=["INVALID_PAIR"],
        )

    notes: list[str] = []
    grouped: dict[str, dict[str, dict[int, InterventionResult]]] = {}
    for result in interventions:
        stage, mode, round_id = (
            result.spec.stage,
            result.spec.mode,
            result.spec.round_id,
        )
        value = result.outcome_metrics.get(metric)
        if not result.valid or value is None or not isfinite(value):
            notes.append(f"INVALID_INTERVENTION:{stage}:{mode}")
            continue
        grouped.setdefault(stage, {}).setdefault(mode, {})[round_id] = result

    if not isfinite(gap):
        notes.append("INSUFFICIENT_FAILURE_GAP")
        return AttributionReport(
            pair=pair,
            outcome="insufficient_failure_gap",
            failure_gap={metric: gap},
            stage_effects={},
            origin_ranking=[],
            origin_set=[],
            roles={},
            notes=notes,
        )

    stage_effects: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    bidirectional: set[str] = set()

    for stage in STAGES:
        results = grouped.get(stage)
        if not results:
            continue

        effect_rounds = {
            round_id
            for mode_results in results.values()
            for round_id in mode_results
        }
        effects: dict[str, float] = {"n_rounds": float(len(effect_rounds))}
        restores = [
            result.outcome_metrics[metric]
            for result in results.get("restore", {}).values()
        ]
        injections = [
            result.outcome_metrics[metric]
            for result in results.get("inject", {}).values()
        ]
        shams = [
            result.outcome_metrics[metric]
            for result in results.get("sham", {}).values()
        ]

        restore_effects = [
            sre(value, m_fail, higher_is_better) for value in restores
        ]
        normalized_restores = [
            value
            for restored in restores
            if (value := nsre(
                restored, m_ref, m_fail, higher_is_better, min_gap
            )) is not None
        ]
        if (value := _mean(restore_effects)) is not None:
            effects["SRE"] = value
        normalized_restore = _mean(normalized_restores)
        if normalized_restore is not None:
            effects["nSRE"] = normalized_restore

        inject_effects = [
            sie(m_ref, value, higher_is_better) for value in injections
        ]
        normalized_injections = [
            value
            for injected in injections
            if (value := nsie(
                m_ref, injected, m_fail, higher_is_better, min_gap
            )) is not None
        ]
        if (value := _mean(inject_effects)) is not None:
            effects["SIE"] = value
        normalized_inject = _mean(normalized_injections)
        if normalized_inject is not None:
            effects["nSIE"] = normalized_inject

        for mode, values in (
            ("restore", normalized_restores),
            ("inject", normalized_injections),
        ):
            if any(value < 0 for value in values) and any(
                value > 0 for value in values
            ):
                notes.append(f"ROUND_SIGN_DISAGREEMENT:{stage}:{mode}")

        combined = bis(normalized_restore, normalized_inject)
        if combined is not None:
            effects["BIS"] = combined
            scores[stage] = combined
            bidirectional.add(stage)
        elif normalized_restore is not None:
            scores[stage] = normalized_restore
        elif normalized_inject is not None:
            scores[stage] = normalized_inject

        sham_deviations = [
            _sham_deviation(value, m_fail, higher_is_better) for value in shams
        ]
        if (value := _mean(sham_deviations)) is not None:
            effects["sham_dev"] = value
            if "SRE" in effects:
                effects["SAE"] = sham_adjusted(effects["SRE"], value)

        if effects.get("nSRE", 0.0) > 1 or effects.get("nSIE", 0.0) > 1:
            notes.append(f"OVERSHOOT:{stage}")
        stage_effects[stage] = effects

    report_args = {
        "pair": pair,
        "failure_gap": {metric: gap},
        "stage_effects": stage_effects,
        "notes": notes,
    }

    if gap < min_gap:
        notes.append("INSUFFICIENT_FAILURE_GAP")
        if gap <= 0:
            notes.append("NONPOSITIVE_FAILURE_GAP")
        return AttributionReport(
            outcome="insufficient_failure_gap",
            origin_ranking=[],
            origin_set=[],
            roles={},
            **report_args,
        )

    sham_violations = [
        stage
        for stage in STAGES
        if "sham_dev" in stage_effects.get(stage, {})
        and abs(stage_effects[stage]["sham_dev"]) >= sham_tolerance
    ]
    if sham_violations:
        notes.extend(f"SHAM_VIOLATION:{stage}" for stage in sham_violations)
        return AttributionReport(
            outcome="sham_violation",
            origin_ranking=[],
            origin_set=[],
            roles={},
            **report_args,
        )

    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    ranking = sorted(
        scores,
        key=lambda stage: (
            -scores[stage],
            stage not in bidirectional,
            stage_order[stage],
        ),
    )
    if ranking:
        top_score = scores[ranking[0]]
        tied = [
            stage
            for stage in ranking
            if abs(top_score - scores[stage]) <= epsilon_tie
        ]
        ranking[: len(tied)] = sorted(
            tied,
            key=lambda stage: (stage not in bidirectional, stage_order[stage]),
        )

    first = pair.first_divergence_stage
    if ranking and first in scores:
        max_score = max(scores.values())
        if max_score - scores[first] <= decisive_margin + epsilon_tie:
            ranking.remove(first)
            ranking.insert(0, first)

    tied = (
        [
            stage
            for stage in ranking
            if abs(scores[ranking[0]] - scores[stage]) <= epsilon_tie
        ]
        if ranking
        else []
    )
    if len(tied) > 1:
        notes.append(f"UNRESOLVED_BETWEEN:{','.join(tied)}")

    for stage in ranking:
        if "sham_dev" not in stage_effects[stage]:
            notes.append(f"SHAM_CONTROL_MISSING:{stage}")
    no_sham_controls = not any(
        "sham_dev" in effects for effects in stage_effects.values()
    )
    if no_sham_controls:
        notes.append("NO_SHAM_CONTROLS")

    all_negligible = bool(ranking) and all(
        abs(scores[stage]) < bystander_threshold for stage in ranking
    )
    unresolved = len(tied) > 1 or no_sham_controls or all_negligible or not ranking
    if unresolved:
        origin_set = tied or ranking
        outcome = "unresolved"
    else:
        origin_set = []
        outcome = "unique_origin"

    roles: dict[str, str] = {}
    for index, stage in enumerate(ranking):
        score = scores[stage]
        if outcome == "unique_origin" and index == 0:
            roles[stage] = "origin_candidate"
        elif stage in origin_set:
            roles[stage] = "unresolved_candidate"
        elif score < 0:
            roles[stage] = "suppressor_candidate"
        elif score < bystander_threshold:
            roles[stage] = "bystander"
        else:
            roles[stage] = "carrier_or_amplifier"

    return AttributionReport(
        outcome=outcome,
        origin_ranking=ranking,
        origin_set=origin_set,
        roles=roles,
        **report_args,
    )
