"""Tests for the intervention engine (Task T5): restore / inject / sham.

Pair design: a reference run and a matched failure run reusing the T4
``compression / aggressive_topk`` failure, active only in round 2 and only on
four of the five selected clients (the fifth is a bystander). Because the
window is a single round, restoring that round's compression state cancels
the failure exactly; the failure itself still leaves a measurable loss gap.
A third run ("other") has a different parameter shape (fewer features) and
fewer recorded rounds, for the shape-mismatch and source-boundary checks.
"""
from types import SimpleNamespace

import pytest

from falcon.intervention import apply_intervention
from falcon.intervention.engine import WARNING_BASE_MODEL_MISMATCH
from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    FailureSpecification,
    InterventionSpecification,
    LocalConfig,
    RunConfig,
    RunMetadata,
    SelectionConfig,
)

ROUNDS = 5
INTERVENTION_ROUND = 2
STAGES = ("selection", "local", "compression", "aggregation", "evaluation")


def _config(run_id: str, failure=None, num_features: int = 20, rounds: int = ROUNDS) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        seed=42,
        rounds=rounds,
        dataset=DatasetConfig(
            num_clients=10,
            num_features=num_features,
            num_classes=2,
            samples_per_client=100,
        ),
        selection=SelectionConfig(clients_per_round=5),
        local=LocalConfig(lr=0.05, local_steps=3, batch_size=32),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
        failure=failure,
    )


def _record(root, cfg: RunConfig) -> Recorder:
    recorder = Recorder(root, cfg.run_id)
    recorder.save_metadata(
        RunMetadata(
            run_id=cfg.run_id,
            seed=cfg.seed,
            rounds=cfg.rounds,
            config=cfg.model_dump(mode="json", exclude={"run_id"}),
            failure=cfg.failure,
        )
    )
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("intervention_runs")
    ref_recorder = _record(root, _config("ref"))
    selected = ref_recorder.load(INTERVENTION_ROUND, "selection").selected_ids
    affected = selected[:4]
    failure = FailureSpecification(
        stage="compression",
        type="aggressive_topk",
        active_rounds=(INTERVENTION_ROUND, INTERVENTION_ROUND),
        severity=1,
        parameters={"k_ratio": 0.02, "affected_clients": affected},
    )
    fail_recorder = _record(root, _config("fail", failure))
    # different update shape (18 vs 42 params) and one fewer recorded round
    _record(root, _config("other", num_features=8, rounds=ROUNDS - 1))
    return SimpleNamespace(
        root=root,
        affected=affected,
        bystander=selected[4],
        ref_final=ref_recorder.load(ROUNDS - 1, "evaluation").metrics,
        fail_final=fail_recorder.load(ROUNDS - 1, "evaluation").metrics,
        ref_round_metrics=ref_recorder.load(INTERVENTION_ROUND, "evaluation").metrics,
        fail_round_metrics=fail_recorder.load(INTERVENTION_ROUND, "evaluation").metrics,
    )


def _spec(**overrides) -> InterventionSpecification:
    kwargs = dict(
        target_run_id="fail",
        source_run_id="ref",
        round_id=INTERVENTION_ROUND,
        stage="compression",
        mode="restore",
    )
    kwargs.update(overrides)
    return InterventionSpecification(**kwargs)


# --- restore / inject / sham behavior -------------------------------------


def test_failure_pair_is_measurably_degraded(pair):
    """Sanity: the single-round failure leaves a real loss gap to recover."""
    assert pair.fail_final["loss"] > pair.ref_final["loss"] + 0.005


def test_restore_at_injected_stage_recovers_reference_metric(pair):
    result = apply_intervention(_spec(), pair.root)

    assert result.valid, result.reason
    assert result.reason is None
    # the single-round window is fully cancelled: exact reference levels
    assert result.outcome_metrics["accuracy"] == pair.ref_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.ref_final["loss"]
    # intervention-round metrics are reported under round_<t>_ keys
    assert result.outcome_metrics["round_2_accuracy"] == pair.ref_round_metrics["accuracy"]
    assert result.outcome_metrics["round_2_loss"] == pair.ref_round_metrics["loss"]
    # same base model at the boundary (pre-divergence): no lineage warning
    assert WARNING_BASE_MODEL_MISMATCH not in result.outcome_metrics


def test_restore_at_pre_divergence_bystander_stage_does_not_recover(pair):
    # "local" at round 2 is before the first divergence (compression), so
    # restoring it is a no-op and the failure plays out unchanged.
    result = apply_intervention(_spec(stage="local"), pair.root)

    assert result.valid, result.reason
    assert result.outcome_metrics == {
        **pair.fail_final,
        "round_2_accuracy": pair.fail_round_metrics["accuracy"],
        "round_2_loss": pair.fail_round_metrics["loss"],
    }
    assert result.outcome_metrics["loss"] != pair.ref_final["loss"]


def test_inject_failed_state_into_reference_degrades_it(pair):
    result = apply_intervention(
        _spec(target_run_id="ref", source_run_id="fail", mode="inject"), pair.root
    )

    assert result.valid, result.reason
    # the reference replays exactly into the failure outcome
    assert result.outcome_metrics["accuracy"] == pair.fail_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.fail_final["loss"]
    assert result.outcome_metrics["loss"] > pair.ref_final["loss"]
    assert WARNING_BASE_MODEL_MISMATCH not in result.outcome_metrics


@pytest.mark.parametrize("stage", STAGES)
def test_sham_at_every_stage_produces_zero_metric_deviation(pair, stage):
    result = apply_intervention(_spec(stage=stage, mode="sham"), pair.root)

    assert result.valid, result.reason
    deviations = {
        key: value
        for key, value in result.outcome_metrics.items()
        if key.startswith("sham_deviation_")
    }
    assert deviations, "sham run must report per-metric deviations"
    assert all(deviation == 0.0 for deviation in deviations.values())
    assert result.outcome_metrics["accuracy"] == pair.fail_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.fail_final["loss"]


# --- validation failure paths (valid=False, never raise) -------------------


def test_missing_target_run_is_invalid(pair):
    result = apply_intervention(_spec(target_run_id="no_such_run"), pair.root)
    assert not result.valid
    assert "target_run_not_found" in result.reason


def test_missing_source_run_is_invalid(pair):
    result = apply_intervention(_spec(source_run_id="no_such_run"), pair.root)
    assert not result.valid
    assert "source_run_not_found" in result.reason


def test_unrecorded_target_boundary_is_invalid(pair):
    result = apply_intervention(_spec(round_id=97), pair.root)
    assert not result.valid
    assert "target_boundary_missing" in result.reason


def test_unrecorded_source_boundary_is_invalid(pair):
    # "other" recorded only rounds 0..ROUNDS-2; the target has ROUNDS rounds
    result = apply_intervention(
        _spec(source_run_id="other", round_id=ROUNDS - 1), pair.root
    )
    assert not result.valid
    assert "source_boundary_missing" in result.reason


def test_shape_mismatch_against_live_state_is_invalid(pair):
    # "other" has 18-parameter updates; the live replay computes 42-parameter ones
    result = apply_intervention(_spec(source_run_id="other"), pair.root)
    assert not result.valid
    assert "shape_mismatch" in result.reason


def test_scoped_client_ids_must_be_present_in_live_and_source(pair):
    result = apply_intervention(_spec(scope={"client_ids": ["client_999"]}), pair.root)
    assert not result.valid
    assert "scoped_clients_missing" in result.reason


def test_client_scope_requires_a_list_stage(pair):
    result = apply_intervention(
        _spec(stage="aggregation", scope={"client_ids": [pair.affected[0]]}), pair.root
    )
    assert not result.valid
    assert "invalid_scope" in result.reason


def test_unknown_scope_keys_are_invalid(pair):
    result = apply_intervention(_spec(scope={"tensor_block": [0]}), pair.root)
    assert not result.valid
    assert "invalid_scope" in result.reason


def test_lineage_mismatch_is_a_warning_not_fatal(pair):
    # at round 3 the failure run's base model has already diverged from the
    # reference's, so the recorded compression lineage cannot match — this
    # must be surfaced as a warning while the intervention proceeds (Plan §13)
    result = apply_intervention(_spec(round_id=3), pair.root)

    assert result.valid, result.reason
    assert result.outcome_metrics[WARNING_BASE_MODEL_MISMATCH] == 1.0
    assert result.outcome_metrics["loss"] != pair.ref_final["loss"]


# --- scoped (partial) replacement, Plan §13.3 ------------------------------


def test_scoped_restore_of_exactly_the_affected_clients_recovers(pair):
    result = apply_intervention(_spec(scope={"client_ids": pair.affected}), pair.root)

    assert result.valid, result.reason
    # only the four affected clients were corrupted: swapping exactly those
    # entries recovers the reference run, identical to a whole-stage restore
    assert result.outcome_metrics["accuracy"] == pair.ref_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.ref_final["loss"]


def test_scoped_restore_of_bystander_client_is_a_noop(pair):
    result = apply_intervention(_spec(scope={"client_ids": [pair.bystander]}), pair.root)

    assert result.valid, result.reason
    # the bystander's state is identical in both runs: replacing it changes nothing
    assert result.outcome_metrics["accuracy"] == pair.fail_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.fail_final["loss"]


def test_partial_scoped_restore_differs_from_both_runs(pair):
    result = apply_intervention(_spec(scope={"client_ids": pair.affected[:1]}), pair.root)

    assert result.valid, result.reason
    # one of four corrupted clients repaired: neither reference nor failure
    assert result.outcome_metrics["loss"] != pair.ref_final["loss"]
    assert result.outcome_metrics["loss"] != pair.fail_final["loss"]
