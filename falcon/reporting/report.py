"""Markdown rendering for attribution reports."""

from __future__ import annotations

from falcon.schema import AttributionReport, FailureSpecification, InterventionResult

_EFFECT_COLUMNS = (
    "SRE",
    "SIE",
    "nSRE",
    "nSIE",
    "BIS",
    "sham_dev",
    "SAE",
    "n_rounds",
    "window",
)


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{100.0 * value:.1f}%"


def render_markdown(
    report: AttributionReport,
    interventions: list[InterventionResult],
    *,
    ground_truth: FailureSpecification | list[FailureSpecification] | None,
) -> str:
    """Render measured evidence, inferred roles, and benchmark truth separately."""
    pair = report.pair
    lines = [
        "# FALCON Attribution Report",
        "",
        "## Pair validity",
        "",
        f"- Reference run: `{pair.reference_run_id}`",
        f"- Failure run: `{pair.failure_run_id}`",
        f"- Status: **{pair.status}**",
        f"- Attribution outcome: **{report.outcome}**",
        f"- First divergence: "
        + (
            f"round {pair.first_divergence_round}, stage `{pair.first_divergence_stage}`"
            if pair.first_divergence_round is not None
            else "not observed"
        ),
        "",
        "| Pair check | Result |",
        "|---|---|",
        *(
            f"| {name} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in pair.checks.items()
        ),
        "",
        "## Terminal failure summary",
        "",
        "| Metric | Failure gap (G) |",
        "|---|---:|",
    ]
    if report.failure_gap:
        lines.extend(
            f"| {metric} | {_number(gap)} |"
            for metric, gap in report.failure_gap.items()
        )
    else:
        lines.append("| — | unavailable |")

    lines.extend(
        [
            "",
            "## Measured evidence — intervention effects",
            "",
            "| Stage | SRE | SIE | nSRE | nSIE | BIS | sham_dev | SAE | n_rounds | window |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if report.stage_effects:
        for stage, effects in report.stage_effects.items():
            values = " | ".join(_number(effects.get(name)) for name in _EFFECT_COLUMNS)
            lines.append(f"| {stage} | {values} |")
    else:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")

    lines.extend(["", "## Inferred origin ranking and roles", ""])
    if report.origin_ranking:
        lines.extend(
            f"{index}. `{stage}` — {report.roles.get(stage, 'unclassified')}"
            for index, stage in enumerate(report.origin_ranking, 1)
        )
    else:
        lines.append("No causal origin ranking is available.")

    lines.extend(["", "## Counterfactual explanation", ""])
    if report.outcome == "unique_origin" and report.origin_ranking:
        origin = report.origin_ranking[0]
        effects = report.stage_effects.get(origin, {})
        lines.append(
            f"Restoring {origin} closes {_percent(effects.get('nSRE'))} of the gap; "
            f"injecting reproduces {_percent(effects.get('nSIE'))}."
        )
        carrier_note = next(
            (note for note in report.notes if note.startswith("CARRIER_TIE_RESOLVED:")),
            None,
        )
        if carrier_note:
            carriers = ", ".join(
                f"`{stage}`" for stage in carrier_note.partition(":")[2].split(",")
            )
            lines.append(
                f"The first-divergent stage `{origin}` is the origin; tied downstream "
                f"stage(s) {carriers} carry or amplify its effect."
            )
    elif "NO_POSITIVE_EVIDENCE_AT_ROUND" in report.notes:
        lines.append(
            "No positive intervention evidence was observed at the analyzed round or window; "
            "attribution remains unresolved."
        )
    elif report.origin_set:
        stages = ", ".join(f"`{stage}`" for stage in report.origin_set)
        lines.append(
            f"Attribution is unresolved among {stages}; no single-stage "
            "counterfactual is supported."
        )
    else:
        lines.append("No counterfactual explanation is supported by the available evidence.")

    invalid = [result for result in interventions if not result.valid]
    lines.extend(
        [
            "",
            "## Warnings and assumptions",
            "",
            "- Effects are conditional on the validated matched execution, deterministic replay, and recorded stage boundaries.",
            f"- Evidence comprises {len(interventions) - len(invalid)} valid of {len(interventions)} attempted interventions.",
        ]
    )
    lines.extend(f"- Pair warning: {warning}" for warning in pair.warnings)
    lines.extend(f"- Analysis note: {note}" for note in report.notes)
    lines.extend(
        f"- Invalid intervention `{result.spec.stage}/{result.spec.mode}`: {result.reason}"
        for result in invalid
    )
    if not pair.warnings and not report.notes and not invalid:
        lines.append("- No additional warnings.")

    if isinstance(ground_truth, list):
        lines.extend(
            [
                "",
                "## Ground truth (benchmark)",
                "",
                "- Injected stages: "
                + ", ".join(f"`{spec.stage}`" for spec in ground_truth),
                "- Injected failure types: "
                + ", ".join(f"`{spec.type}`" for spec in ground_truth),
                "- Predicted origin ranking: "
                + (", ".join(f"`{stage}`" for stage in report.origin_ranking) or "unresolved"),
                "- Predicted origin set: "
                + (", ".join(f"`{stage}`" for stage in report.origin_set) or "none"),
            ]
        )
    elif ground_truth is not None:
        resolved = report.outcome == "unique_origin" and bool(report.origin_ranking)
        predicted = report.origin_ranking[0] if resolved else "unresolved"
        verdict = "yes" if predicted == ground_truth.stage else "no"
        if not resolved:
            verdict = "unresolved"
        lines.extend(
            [
                "",
                "## Ground truth (benchmark)",
                "",
                f"- Injected stage: `{ground_truth.stage}`",
                f"- Injected failure type: `{ground_truth.type}`",
                f"- Predicted origin: `{predicted}`",
                f"- Prediction matches injected stage: **{verdict}**",
            ]
        )

    return "\n".join(lines) + "\n"
