"""Tests for the intervention engine (Task T5): restore / inject / sham.

Pair design: a reference run and a matched failure run reusing the T4
``compression / aggressive_topk`` failure, active only in round 2 and only on
four of the five selected clients (the fifth is a bystander). Because the
window is a single round, restoring that round's compression state cancels
the failure exactly; the failure itself still leaves a measurable loss gap.
A third run ("other") has a different dataset layout (fewer features) and
fewer recorded rounds, for the incompatible-runs checks. Crafted copies of
the pair runs (same metadata, tampered recordings) feed the validation-failure
paths — after T5-F finding 4, a cross-run transplant is gated on compatible
metadata, so shape/dtype/finiteness defects must be smuggled in inside a
COMPATIBLE recording.
"""
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from falcon.intervention import apply_intervention
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
#: stage -> array field the overlay checks against the live state
_ARRAY_FIELD = {"local": "update", "compression": "update", "aggregation": "aggregate"}
#: (target, template the crafted source is copied from, mode) — both directions
_DIRECTIONS = (
    ("fail", "ref", "restore"),  # reference state into the failure run
    ("ref", "fail", "inject"),  # failure state into the reference run
)


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
    # different dataset layout (18 vs 42 params) and one fewer recorded round
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


def _craft_run(pair, template: str, name: str):
    """Copy a recorded run (metadata included) under a new run id."""
    shutil.copytree(pair.root / "runs" / template, pair.root / "runs" / name)
    return name


def _rewrite_boundary(pair, run_id: str, stage: str, mutate):
    """Re-record one boundary of a crafted run with ``mutate`` applied."""
    recorder = Recorder(pair.root, run_id)
    state = recorder.load(INTERVENTION_ROUND, stage)
    recorder.record(INTERVENTION_ROUND, stage, mutate(state))


def _defective(array: np.ndarray, defect: str) -> np.ndarray:
    if defect == "shape":
        return np.zeros(array.shape[0] + 1, dtype=array.dtype)
    if defect == "dtype":
        return array.astype(np.float32)
    if defect == "non_finite":
        bad = np.array(array, copy=True)
        bad.flat[0] = float("nan")
        return bad
    raise AssertionError(f"unknown defect {defect!r}")


def _array_defect(stage: str, defect: str):
    """Mutate the checked array field of every state in a boundary."""

    def apply(state):
        field = _ARRAY_FIELD[stage]
        if isinstance(state, list):
            return [
                entry.model_copy(update={field: _defective(getattr(entry, field), defect)})
                for entry in state
            ]
        return state.model_copy(update={field: _defective(getattr(state, field), defect)})

    return apply


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


def test_restore_at_pre_divergence_bystander_stage_does_not_recover(pair):
    # "local" at round 2 is before the first divergence (compression), so
    # restoring it is a no-op and the failure plays out unchanged.
    result = apply_intervention(_spec(stage="local"), pair.root)

    assert result.valid, result.reason
    # outcome_metrics also carries flattened class_<c>_* entries; the failure
    # metrics must replay unchanged
    expected = {
        **pair.fail_final,
        "round_2_accuracy": pair.fail_round_metrics["accuracy"],
        "round_2_loss": pair.fail_round_metrics["loss"],
    }
    assert expected.items() <= result.outcome_metrics.items()
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


@pytest.mark.parametrize("stage", STAGES)
def test_sham_rejects_drifted_replay(pair, stage):
    """Finding-2 regression: the sham gate must not certify a drifting replay.

    Only the recorded metadata is drifted (local.lr 0.05 -> 50.0); the
    recording itself is untouched. The no-overlay replay can no longer
    reproduce the recorded boundary hashes, so the sham must REJECT — the
    pre-fix sham overlaid the recorded boundary and reported deviation 0.0.
    """
    name = _craft_run(pair, "fail", f"drifted_{stage}")
    metadata_path = pair.root / "runs" / name / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["local"]["lr"] = 50.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = apply_intervention(
        _spec(target_run_id=name, stage=stage, mode="sham"), pair.root
    )

    assert not result.valid
    assert result.reason.startswith("replay_drift:")


# --- validation failure paths (valid=False, never raise) -------------------


def test_missing_target_run_is_invalid(pair):
    result = apply_intervention(_spec(target_run_id="no_such_run"), pair.root)
    assert not result.valid
    assert "target_run_not_found" in result.reason


def test_missing_source_run_is_invalid(pair):
    result = apply_intervention(_spec(source_run_id="no_such_run"), pair.root)
    assert not result.valid
    assert "source_run_not_found" in result.reason


@pytest.mark.parametrize("round_id", [-1, ROUNDS, 97])
def test_out_of_range_round_is_invalid(pair, round_id):
    result = apply_intervention(_spec(round_id=round_id), pair.root)
    assert not result.valid
    assert "invalid_round_id" in result.reason


def test_unrecorded_target_boundary_is_invalid(pair):
    # metadata claims ROUNDS rounds but the last one was never recorded
    name = _craft_run(pair, "fail", "gap_target")
    shutil.rmtree(pair.root / "runs" / name / f"round_{ROUNDS - 1}")
    result = apply_intervention(_spec(target_run_id=name, round_id=ROUNDS - 1), pair.root)
    assert not result.valid
    assert "target_boundary_missing" in result.reason


def test_unrecorded_source_boundary_is_invalid(pair):
    # a COMPATIBLE source (same metadata) whose recording stops one round early
    name = _craft_run(pair, "fail", "gap_source")
    shutil.rmtree(pair.root / "runs" / name / f"round_{ROUNDS - 1}")
    result = apply_intervention(
        _spec(source_run_id=name, round_id=ROUNDS - 1), pair.root
    )
    assert not result.valid
    assert "source_boundary_missing" in result.reason


def test_incompatible_dataset_layout_is_invalid(pair):
    # finding 4: "other" flattens to a different parameter count; even equal
    # flat lengths would not imply compatible coordinate meaning
    result = apply_intervention(_spec(source_run_id="other"), pair.root)
    assert not result.valid
    assert "incompatible_runs" in result.reason


def test_incompatible_seed_is_invalid(pair):
    name = _craft_run(pair, "ref", "otherseed")
    metadata_path = pair.root / "runs" / name / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] += 1
    metadata["config"]["seed"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = apply_intervention(_spec(source_run_id=name), pair.root)
    assert not result.valid
    assert "incompatible_runs" in result.reason


def test_incompatible_config_delta_beyond_failure_is_invalid(pair):
    name = _craft_run(pair, "ref", "otherlr")
    metadata_path = pair.root / "runs" / name / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"]["local"]["lr"] = 0.5  # delta beyond the failure key
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = apply_intervention(_spec(source_run_id=name), pair.root)
    assert not result.valid
    assert "incompatible_runs" in result.reason


@pytest.mark.parametrize("defect", ["shape", "dtype", "non_finite"])
@pytest.mark.parametrize("stage", sorted(_ARRAY_FIELD))
@pytest.mark.parametrize("direction", _DIRECTIONS, ids=["restore", "inject"])
def test_array_defects_are_invalid_per_stage_and_direction(
    pair, direction, stage, defect
):
    """Shape/dtype/finiteness checks fire per array stage, both directions."""
    target, template, mode = direction
    name = _craft_run(pair, template, f"bad_{defect}_{stage}_{mode}")
    _rewrite_boundary(pair, name, stage, _array_defect(stage, defect))

    result = apply_intervention(
        _spec(target_run_id=target, source_run_id=name, stage=stage, mode=mode),
        pair.root,
    )

    expected_reason = {
        "shape": "shape_mismatch",
        "dtype": "dtype_mismatch",
        "non_finite": "non_finite_state",
    }[defect]
    assert not result.valid
    assert expected_reason in result.reason


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


@pytest.mark.parametrize(
    "bad_scope",
    [None, "client_0", [], ["client_0", "client_0"], [""], ["client_0", 7]],
    ids=["none", "string", "empty", "duplicate", "empty_string", "non_string"],
)
def test_invalid_client_scope_values_are_invalid(pair, bad_scope):
    result = apply_intervention(_spec(scope={"client_ids": bad_scope}), pair.root)
    assert not result.valid
    assert "invalid_scope" in result.reason


def test_corrupt_recorded_hash_is_invalid(pair):
    name = _craft_run(pair, "fail", "corrupt")
    path = (
        pair.root / "runs" / name / f"round_{INTERVENTION_ROUND}" / "aggregation.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["received_ids"].append("client_tampered")  # content_hash now stale
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = apply_intervention(
        _spec(target_run_id=name, stage="aggregation"), pair.root
    )
    assert not result.valid
    assert "target_boundary_invalid" in result.reason


def test_wrong_state_type_is_invalid(pair):
    # local.json holding a valid AggregationState: loads fine, wrong type
    name = _craft_run(pair, "fail", "wrongtype")
    round_dir = pair.root / "runs" / name / f"round_{INTERVENTION_ROUND}"
    shutil.rmtree(round_dir / "local")
    shutil.copy(round_dir / "aggregation.json", round_dir / "local.json")
    shutil.copy(round_dir / "aggregation.npz", round_dir / "local.npz")

    result = apply_intervention(_spec(target_run_id=name, stage="local"), pair.root)
    assert not result.valid
    assert "target_state_type_mismatch" in result.reason


def test_rogue_recorded_round_is_invalid(pair):
    # a recording stored under round 2 whose payload claims round 99
    name = _craft_run(pair, "fail", "rogueround")
    recorder = Recorder(pair.root, name)
    state = recorder.load(INTERVENTION_ROUND, "aggregation")
    recorder.record(
        INTERVENTION_ROUND, "aggregation", state.model_copy(update={"round_id": 99})
    )

    result = apply_intervention(
        _spec(target_run_id=name, stage="aggregation"), pair.root
    )
    assert not result.valid
    assert "target_round_mismatch" in result.reason


def test_lineage_mismatch_is_invalid(pair):
    # finding 3: at round 3 the failure run's base model has already diverged
    # from the reference's, so the recorded compression lineage cannot match —
    # this is REJECTED, not downgraded to a warning
    result = apply_intervention(_spec(round_id=3), pair.root)

    assert not result.valid
    assert "lineage_mismatch" in result.reason


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
