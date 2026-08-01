import pytest

from falcon.metrics.effects import (
    bis,
    failure_gap,
    nsie,
    nsre,
    sham_adjusted,
    sie,
    sre,
)


@pytest.mark.parametrize(
    ("higher_is_better", "m_ref", "m_fail", "expected"),
    [(True, 0.9, 0.5, 0.4), (False, 0.2, 0.6, 0.4)],
)
def test_failure_gap_respects_metric_direction(
    higher_is_better, m_ref, m_fail, expected
):
    assert failure_gap(m_ref, m_fail, higher_is_better) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("higher_is_better", "restored", "failed", "expected"),
    [(True, 0.8, 0.5, 0.3), (False, 0.3, 0.6, 0.3)],
)
def test_sre_respects_metric_direction(
    higher_is_better, restored, failed, expected
):
    assert sre(restored, failed, higher_is_better) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("higher_is_better", "reference", "injected", "expected"),
    [(True, 0.9, 0.6, 0.3), (False, 0.2, 0.5, 0.3)],
)
def test_sie_respects_metric_direction(
    higher_is_better, reference, injected, expected
):
    assert sie(reference, injected, higher_is_better) == pytest.approx(expected)


def test_normalized_effects_are_hand_computed():
    assert nsre(0.8, 0.9, 0.5) == pytest.approx(0.75)
    assert nsie(0.9, 0.7, 0.5) == pytest.approx(0.5)
    assert nsre(0.3, 0.2, 0.6, higher_is_better=False) == pytest.approx(0.75)
    assert nsie(0.2, 0.4, 0.6, higher_is_better=False) == pytest.approx(0.5)


def test_nsre_preserves_above_one_and_negative_values():
    assert nsre(1.0, 0.9, 0.5) == pytest.approx(1.25)
    assert nsre(0.4, 0.9, 0.5) == pytest.approx(-0.25)


def test_normalized_effects_refuse_too_small_a_gap():
    assert nsre(0.5, 0.5 + 1e-10, 0.5) is None
    assert nsie(0.5 + 1e-10, 0.5, 0.5) is None
    assert nsre(0.51, 0.51, 0.5, min_gap=0.02) is None
    assert nsie(0.51, 0.5, 0.5, min_gap=0.02) is None


def test_bidirectional_intervention_score():
    assert bis(0.8, 0.6) == pytest.approx(0.6)
    assert bis(0.8, 0.6, lam=0.25) == pytest.approx(0.65)
    assert bis(None, 0.6) is None
    assert bis(0.8, None) is None


def test_sham_adjusted_effect():
    assert sham_adjusted(0.35, 0.05) == pytest.approx(0.3)
