import pytest

from falcon.attribution.analyzer import attribute
from falcon.schema import (
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


def _result(stage, mode, value=0.0, *, valid=True, reason=None, metric="accuracy"):
    target, source = (
        ("failure", "reference") if mode != "inject" else ("reference", "failure")
    )
    return InterventionResult(
        spec=InterventionSpecification(
            target_run_id=target,
            source_run_id=source,
            round_id=1,
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
        ],
        epsilon_tie=0.1,
    )

    assert report.stage_effects["aggregation"]["BIS"] > report.stage_effects["local"]["BIS"]
    assert report.origin_ranking[:2] == ["local", "aggregation"]
    assert report.roles["local"] == "origin_candidate"
    assert report.roles["aggregation"] == "carrier_or_amplifier"


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
    assert report.roles == {}
    assert "INSUFFICIENT_FAILURE_GAP" in report.notes
    assert "nSRE" not in report.stage_effects["local"]


def test_tied_scores_are_reported_as_unresolved():
    report = _attribute(
        _pair("selection"),
        [
            _result("local", "restore", 0.82),
            _result("compression", "inject", 0.58),
        ],
    )

    assert report.origin_ranking[:2] == ["local", "compression"]
    assert "UNRESOLVED_BETWEEN:local,compression" in report.notes


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
