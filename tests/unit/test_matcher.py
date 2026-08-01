import json

import numpy as np

from falcon.matcher.matcher import validate_pair
from falcon.recorder import Recorder
from falcon.schema import (
    AggregationState,
    ClientLocalState,
    CompressionState,
    FailureSpecification,
    OutcomeState,
    RunMetadata,
    SelectionState,
)


def _failure() -> FailureSpecification:
    return FailureSpecification(
        stage="local", type="test_failure", active_rounds=(1, 1)
    )


def _config(failure=None):
    return {
        "dataset": {"name": "synthetic", "seed": 1001},
        "local": {"lr": 0.1},
        "failure": failure.model_dump(mode="json") if failure else None,
    }


def _states(round_id: int):
    update = np.array([round_id + 0.25, -0.5], dtype=np.float64)
    return {
        "selection": SelectionState(
            round_id=round_id,
            candidate_ids=["c0"],
            selected_ids=["c0"],
            sampling_probs={"c0": 1.0},
        ),
        "local": [
            ClientLocalState(
                round_id=round_id,
                client_id="c0",
                base_model_hash="base",
                update=update,
                num_examples=4,
                num_steps=1,
                loss_history=[0.5],
            )
        ],
        "compression": [
            CompressionState(
                round_id=round_id,
                client_id="c0",
                uncompressed_hash="update",
                update=update.copy(),
            )
        ],
        "aggregation": AggregationState(
            round_id=round_id,
            received_ids=["c0"],
            accepted_ids=["c0"],
            rejected_ids=[],
            weights={"c0": 1.0},
            aggregate=update.copy(),
        ),
        "evaluation": OutcomeState(
            round_id=round_id,
            model_hash=f"model-{round_id}",
            metrics={"loss": 0.5},
        ),
    }


def _make_pair(tmp_path, divergent=True):
    failure = _failure()
    reference = Recorder(tmp_path, "reference")
    failed = Recorder(tmp_path, "failure")
    reference.save_metadata(
        RunMetadata(
            run_id="reference",
            seed=7,
            rounds=2,
            config=_config(),
        )
    )
    failed.save_metadata(
        RunMetadata(
            run_id="failure",
            seed=7,
            rounds=2,
            config=_config(failure),
            failure=failure,
        )
    )
    for round_id in range(2):
        for stage, state in _states(round_id).items():
            reference.record(round_id, stage, state)
            failed.record(round_id, stage, _states(round_id)[stage])
    if divergent:
        local = failed.load(1, "local")
        local[0].update[0] += 1.0
        failed.record(1, "local", local)
    return reference.run_dir, failed.run_dir


def _rewrite_metadata(run_dir, update):
    path = run_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_matched_pair(tmp_path):
    reference, failure = _make_pair(tmp_path)

    report = validate_pair(reference, failure)

    assert report.status == "MATCHED"
    assert all(report.checks.values())
    assert (report.first_divergence_round, report.first_divergence_stage) == (
        1,
        "local",
    )


def test_corrupted_seed_is_invalid(tmp_path):
    reference, failure = _make_pair(tmp_path)
    _rewrite_metadata(failure, lambda data: data.update(seed=8))

    report = validate_pair(reference, failure)

    assert report.status == "INVALID_PAIR"
    assert not report.checks["same_seed"]


def test_extra_config_difference_is_invalid(tmp_path):
    reference, failure = _make_pair(tmp_path)
    _rewrite_metadata(failure, lambda data: data["config"]["local"].update(lr=0.2))

    report = validate_pair(reference, failure)

    assert report.status == "INVALID_PAIR"
    assert not report.checks["config_delta_is_failure_only"]


def test_pre_failure_hash_mismatch_is_invalid(tmp_path):
    reference, failure = _make_pair(tmp_path)
    outcome = Recorder(tmp_path, "failure").load(0, "evaluation")
    outcome.metrics["loss"] = 0.75
    Recorder(tmp_path, "failure").record(0, "evaluation", outcome)

    report = validate_pair(reference, failure)

    assert report.status == "INVALID_PAIR"
    assert not report.checks["pre_failure_hashes_match"]
    assert (report.first_divergence_round, report.first_divergence_stage) == (
        0,
        "evaluation",
    )


def test_identical_runs_warn(tmp_path):
    reference, failure = _make_pair(tmp_path, divergent=False)

    report = validate_pair(reference, failure)

    assert report.status == "MATCHED_WITH_WARNINGS"
    assert report.first_divergence_round is None
    assert report.first_divergence_stage is None
    assert report.warnings == ["runs are identical - no failure effect recorded"]
