import pytest

from falcon.attribution.analyzer import attribute
from falcon.reporting.report import render_markdown
from falcon.schema import (
    FailureSpecification,
    InterventionResult,
    InterventionSpecification,
    PairValidationReport,
)


def _pair(first_stage="local"):
    return PairValidationReport(
        reference_run_id="reference",
        failure_run_id="failure",
        status="MATCHED",
        checks={"matched": True},
        first_divergence_round=1,
        first_divergence_stage=first_stage,
    )


def _result(
    stage,
    mode,
    value=0.0,
    *,
    valid=True,
    reason=None,
    metric="accuracy",
    round_id=1,
    round_window=None,
):
    target, source = (
        ("failure", "reference") if mode != "inject" else ("reference", "failure")
    )
    return InterventionResult(
        spec=InterventionSpecification(
            target_run_id=target,
            source_run_id=source,
            round_id=round_id,
            round_window=round_window,
            stage=stage,
            mode=mode,
        ),
        valid=valid,
        reason=reason,
        outcome_metrics={metric: value},
    )


def _attribute(pair, interventions, **kwargs):
    return attribute(
        pair,
        interventions,
        metric="accuracy",
        m_ref=0.9,
        m_fail=0.5,
        min_gap=0.01,
        sham_tolerance=0.01,
        **kwargs,
    )


@pytest.fixture
def selection_aggregation_carrier_tie():
    window = (1, 3)
    return _pair("selection"), [
        _result("selection", "restore", 0.82, round_window=window),
        _result("selection", "inject", 0.58, round_window=window),
        _result("selection", "sham", 0.5, round_window=window),
        _result("aggregation", "restore", 0.82, round_window=window),
        _result("aggregation", "inject", 0.58, round_window=window),
    ]


@pytest.fixture
def zero_tie_with_negative_compression():
    return _pair("compression"), [
        _result("selection", "restore", 0.5),
        _result("selection", "inject", 0.9),
        _result("local", "restore", 0.5),
        _result("local", "inject", 0.9),
        _result("compression", "restore", 0.4),
        _result("compression", "inject", 1.0),
        _result("compression", "sham", 0.5),
    ]


@pytest.fixture
def carrier_tie_with_upstream_stage():
    return _pair("local"), [
        _result("selection", "restore", 0.82),
        _result("selection", "inject", 0.58),
        _result("local", "restore", 0.82),
        _result("local", "inject", 0.58),
        _result("local", "sham", 0.5),
        _result("aggregation", "restore", 0.82),
        _result("aggregation", "inject", 0.58),
    ]


def test_downstream_carrier_tie_resolves_to_first_divergence(
    selection_aggregation_carrier_tie,
):
    pair, interventions = selection_aggregation_carrier_tie
    report = _attribute(pair, interventions)

    assert report.outcome == "unique_origin"
    assert report.origin_ranking[:2] == ["selection", "aggregation"]
    assert report.roles["selection"] == "origin_candidate"
    assert report.roles["aggregation"] == "carrier_or_amplifier"
    assert "CARRIER_TIE_RESOLVED:aggregation" in report.notes
    assert report.stage_effects["selection"]["window"] == 1.0

    markdown = render_markdown(
        report,
        interventions,
        ground_truth=FailureSpecification(
            stage="selection", type="test", active_rounds=(1, 3)
        ),
    )
    assert "tied downstream stage(s) `aggregation` carry or amplify" in markdown
    assert "Prediction matches injected stage: **yes**" in markdown


def test_zero_effect_bystanders_rank_below_negative_material_evidence(
    zero_tie_with_negative_compression,
):
    pair, interventions = zero_tie_with_negative_compression
    report = _attribute(pair, interventions)

    assert report.outcome == "unresolved"
    assert report.origin_ranking == ["compression", "selection", "local"]
    assert report.roles["selection"] == report.roles["local"] == "bystander"
    assert "NO_POSITIVE_EVIDENCE_AT_ROUND" in report.notes
    assert "UNRESOLVED_BETWEEN:selection,local" not in report.notes

    markdown = render_markdown(report, interventions, ground_truth=None)
    assert "No positive intervention evidence was observed" in markdown


def test_carrier_tie_with_non_downstream_stage_stays_unresolved(
    carrier_tie_with_upstream_stage,
):
    pair, interventions = carrier_tie_with_upstream_stage
    report = _attribute(pair, interventions)

    assert report.outcome == "unresolved"
    assert report.origin_set == ["local", "selection", "aggregation"]
    assert "UNRESOLVED_BETWEEN:local,selection,aggregation" in report.notes
    assert not any(note.startswith("CARRIER_TIE_RESOLVED:") for note in report.notes)


def test_clean_single_stage_attribution():
    report = _attribute(
        _pair("local"),
        [
            _result("selection", "restore", 0.54),
            _result("local", "restore", 0.86),
            _result("local", "inject", 0.54),
            _result("local", "sham", 0.5),
            _result("aggregation", "restore", 0.7),
        ],
    )

    assert report.failure_gap == {"accuracy": pytest.approx(0.4)}
    assert report.outcome == "unique_origin"
    assert report.origin_set == []
    assert report.origin_ranking == ["local", "aggregation", "selection"]
    assert report.roles == {
        "local": "origin_candidate",
        "aggregation": "carrier_or_amplifier",
        "selection": "carrier_or_amplifier",
    }
    assert report.stage_effects["local"] == pytest.approx(
        {
            "SRE": 0.36,
            "SIE": 0.36,
            "nSRE": 0.9,
            "nSIE": 0.9,
            "BIS": 0.9,
            "sham_dev": 0.0,
            "SAE": 0.36,
            "n_rounds": 1.0,
        }
    )


def test_first_divergence_wins_downstream_restoration_trap():
    report = _attribute(
        _pair("local"),
        [
            _result("local", "restore", 0.86),
            _result("local", "inject", 0.54),
            _result("aggregation", "restore", 0.88),
            _result("aggregation", "inject", 0.52),
            _result("local", "sham", 0.5),
        ],
    )

    assert report.stage_effects["aggregation"]["BIS"] > report.stage_effects["local"]["BIS"]
    assert report.origin_ranking[:2] == ["local", "aggregation"]
    assert report.roles["local"] == "origin_candidate"
    assert report.roles["aggregation"] == "carrier_or_amplifier"
    assert report.outcome == "unique_origin"


def test_sham_violation_kills_whole_report():
    report = _attribute(
        _pair("compression"),
        [
            _result("compression", "restore", 0.86),
            _result("compression", "sham", 0.53),
            _result("aggregation", "restore", 0.8),
        ],
    )

    assert report.origin_ranking == []
    assert report.outcome == "sham_violation"
    assert report.roles == {}
    assert "SHAM_VIOLATION:compression" in report.notes
    assert report.stage_effects["compression"]["sham_dev"] == pytest.approx(0.03)


def test_insufficient_gap_has_no_ranking():
    report = attribute(
        _pair("local"),
        [_result("local", "restore", 0.5004)],
        metric="accuracy",
        m_ref=0.5005,
        m_fail=0.5,
        min_gap=0.01,
        sham_tolerance=0.01,
    )

    assert report.origin_ranking == []
    assert report.outcome == "insufficient_failure_gap"
    assert report.roles == {}
    assert "INSUFFICIENT_FAILURE_GAP" in report.notes
    assert "nSRE" not in report.stage_effects["local"]


def test_tied_scores_are_reported_as_unresolved():
    report = _attribute(
        _pair("selection"),
        [
            _result("local", "restore", 0.82),
            _result("compression", "inject", 0.58),
            _result("aggregation", "restore", 0.82),
            _result("local", "sham", 0.5),
        ],
    )

    assert report.outcome == "unresolved"
    assert report.origin_set == ["local", "compression", "aggregation"]
    assert report.origin_ranking[:3] == report.origin_set
    assert not any(role == "origin_candidate" for role in report.roles.values())
    assert "UNRESOLVED_BETWEEN:local,compression,aggregation" in report.notes


def test_invalid_interventions_are_excluded_and_named():
    report = _attribute(
        _pair("local"),
        [
            _result(
                "selection",
                "restore",
                valid=False,
                reason="shape_mismatch",
            ),
            _result("local", "restore", 0.86),
        ],
    )

    assert report.origin_ranking == ["local"]
    assert "selection" not in report.stage_effects
    assert "INVALID_INTERVENTION:selection:restore" in report.notes


def test_first_divergence_promotion_beats_pipeline_order_tie_sort():
    report = _attribute(
        _pair("local"),
        [
            _result("selection", "restore", 0.86),
            _result("selection", "inject", 0.54),
            _result("local", "restore", 0.86),
            _result("local", "inject", 0.54),
            _result("local", "sham", 0.5),
        ],
    )

    assert report.origin_ranking[:2] == ["local", "selection"]
    assert report.outcome == "unresolved"
    assert report.origin_set == ["local", "selection"]


@pytest.mark.parametrize("m_ref", [0.5, float("nan")])
def test_nonpositive_or_nonfinite_gap_is_insufficient(m_ref):
    report = attribute(
        _pair("local"),
        [_result("local", "restore", 0.85), _result("local", "sham", 0.9)],
        metric="accuracy",
        m_ref=m_ref,
        m_fail=0.9,
        min_gap=0.01,
        sham_tolerance=0.01,
    )

    assert report.outcome == "insufficient_failure_gap"
    assert report.origin_ranking == []
    assert "INSUFFICIENT_FAILURE_GAP" in report.notes
    if m_ref == 0.5:
        assert "NONPOSITIVE_FAILURE_GAP" in report.notes


def test_nonfinite_intervention_metric_is_rejected():
    report = _attribute(
        _pair("local"),
        [_result("local", "restore", float("nan")), _result("local", "sham", 0.5)],
    )

    assert "INVALID_INTERVENTION:local:restore" in report.notes
    assert "nSRE" not in report.stage_effects["local"]
    assert report.outcome == "unresolved"


def test_multi_round_effects_are_means_and_report_round_count():
    report = _attribute(
        _pair("local"),
        [
            _result("local", "restore", 0.55, round_id=1),
            _result("local", "restore", 0.85, round_id=2),
            _result("local", "inject", 0.85, round_id=1),
            _result("local", "inject", 0.55, round_id=2),
            _result("local", "sham", 0.5, round_id=1),
            _result("local", "sham", 0.5, round_id=2),
        ],
    )

    assert report.stage_effects["local"] == pytest.approx(
        {
            "n_rounds": 2,
            "SRE": 0.2,
            "nSRE": 0.5,
            "SIE": 0.2,
            "nSIE": 0.5,
            "BIS": 0.5,
            "sham_dev": 0.0,
            "SAE": 0.2,
        }
    )


def test_multi_round_sign_disagreement_is_noted():
    report = _attribute(
        _pair("local"),
        [
            _result("local", "restore", 0.45, round_id=1),
            _result("local", "restore", 0.85, round_id=2),
            _result("local", "sham", 0.5, round_id=1),
        ],
    )

    assert "ROUND_SIGN_DISAGREEMENT:local:restore" in report.notes


@pytest.mark.parametrize("sham_valid", [None, False])
def test_no_valid_sham_controls_make_outcome_unresolved(sham_valid):
    interventions = [_result("local", "restore", 0.86)]
    if sham_valid is False:
        interventions.append(
            _result("local", "sham", 0.5, valid=False, reason="replay_drift")
        )

    report = _attribute(_pair("local"), interventions)

    assert report.outcome == "unresolved"
    assert report.origin_set == ["local"]
    assert "SHAM_CONTROL_MISSING:local" in report.notes
    assert "NO_SHAM_CONTROLS" in report.notes
    assert report.roles["local"] != "origin_candidate"


def test_sham_tolerance_boundary_is_a_violation():
    report = _attribute(
        _pair("local"),
        [_result("local", "restore", 0.86), _result("local", "sham", 0.51)],
    )

    assert report.outcome == "sham_violation"
    assert "SHAM_VIOLATION:local" in report.notes


def test_overshoot_and_sham_adjusted_effect_are_reported():
    report = _attribute(
        _pair("local"),
        [
            _result("local", "restore", 1.0),
            _result("local", "inject", 0.4),
            _result("local", "sham", 0.5),
        ],
    )

    assert report.stage_effects["local"]["nSRE"] == pytest.approx(1.25)
    assert report.stage_effects["local"]["SAE"] == pytest.approx(0.5)
    assert "OVERSHOOT:local" in report.notes


def test_equal_score_prefers_bidirectional_evidence():
    report = _attribute(
        _pair("aggregation"),
        [
            _result("selection", "restore", 0.82),
            _result("local", "restore", 0.82),
            _result("local", "inject", 0.58),
            _result("local", "sham", 0.5),
        ],
    )

    assert report.origin_ranking[:2] == ["local", "selection"]


def test_all_negligible_effects_do_not_fabricate_an_origin():
    report = _attribute(
        _pair("local"),
        [_result("local", "restore", 0.52), _result("local", "sham", 0.5)],
    )

    assert report.outcome == "unresolved"
    assert report.roles["local"] != "origin_candidate"


def test_unresolved_report_suppresses_single_stage_counterfactual_and_verdict():
    report = _attribute(
        _pair("selection"),
        [
            _result("local", "restore", 0.82),
            _result("compression", "inject", 0.58),
            _result("local", "sham", 0.5),
        ],
    )
    ground_truth = FailureSpecification(
        stage="compression", type="test", active_rounds=(1, 1)
    )

    markdown = render_markdown(report, [], ground_truth=ground_truth)

    assert "Restoring local closes" not in markdown
    assert "unresolved among `local`, `compression`" in markdown
    assert "Prediction matches injected stage: **unresolved**" in markdown
