"""Tests for windowed interventions (Task T13, Plan §13.5).

Pair design: same reference/failure pair as the T5 engine tests, but the T4
``compression / aggressive_topk`` failure is active over a two-round window
(rounds 2–3) instead of a single round, on four of the five selected clients.
A single-round restore at the first active round leaves round 3 corrupted and
stays degraded; the windowed restore cancels the failure exactly. Both runs
replay deterministically from the same seed, so exact metric equality with
the reference/failure recording is the oracle for a full cancel/reproduction.
"""
import json
import shutil
from types import SimpleNamespace

import pytest

from falcon.intervention import apply_intervention
from falcon.intervention.__main__ import main as cli_main
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
WINDOW = (2, 3)  # inclusive failure window [t1, t2]


def _config(run_id: str, failure=None, rounds: int = ROUNDS) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        seed=42,
        rounds=rounds,
        dataset=DatasetConfig(
            num_clients=10,
            num_features=20,
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
    root = tmp_path_factory.mktemp("window_runs")
    ref_recorder = _record(root, _config("ref"))
    selected = ref_recorder.load(WINDOW[0], "selection").selected_ids
    failure = FailureSpecification(
        stage="compression",
        type="aggressive_topk",
        active_rounds=WINDOW,
        severity=1,
        parameters={"k_ratio": 0.02, "affected_clients": selected[:4]},
    )
    fail_recorder = _record(root, _config("fail", failure))
    return SimpleNamespace(
        root=root,
        ref_final=ref_recorder.load(ROUNDS - 1, "evaluation").metrics,
        fail_final=fail_recorder.load(ROUNDS - 1, "evaluation").metrics,
        ref_round_metrics={
            t: ref_recorder.load(t, "evaluation").metrics for t in WINDOW
        },
    )


def _spec(**overrides) -> InterventionSpecification:
    kwargs = dict(
        target_run_id="fail",
        source_run_id="ref",
        round_id=WINDOW[0],  # ignored when round_window is set
        stage="compression",
        mode="restore",
    )
    kwargs.update(overrides)
    return InterventionSpecification(**kwargs)


def _craft_run(pair, template: str, name: str):
    """Copy a recorded run (metadata included) under a new run id."""
    shutil.copytree(pair.root / "runs" / template, pair.root / "runs" / name)
    return name


# --- windowed restore / inject / sham behavior ----------------------------


def test_failure_window_is_measurably_degraded(pair):
    """Sanity: the two-round failure leaves a real loss gap to recover."""
    assert pair.fail_final["loss"] > pair.ref_final["loss"] + 0.005


def test_windowed_restore_recovers_better_than_single_round(pair):
    """The T13 motivation, both directions on the same pair.

    Single-round restore at the first active round leaves round 3 failing;
    the windowed restore covers every active round and cancels the failure
    exactly. The windowed outcome must be strictly better AND at reference
    level, the single-round outcome strictly worse AND not at reference level.
    """
    single = apply_intervention(_spec(round_id=WINDOW[0]), pair.root)
    windowed = apply_intervention(_spec(round_window=WINDOW), pair.root)

    assert single.valid, single.reason
    assert windowed.valid, windowed.reason
    # windowed restore: exact reference levels (full cancel)
    assert windowed.outcome_metrics["accuracy"] == pair.ref_final["accuracy"]
    assert windowed.outcome_metrics["loss"] == pair.ref_final["loss"]
    # single-round restore: still degraded, strictly worse than the window
    assert single.outcome_metrics["loss"] != pair.ref_final["loss"]
    assert single.outcome_metrics["loss"] > windowed.outcome_metrics["loss"]


def test_windowed_restore_reports_t1_and_t2_metrics(pair):
    result = apply_intervention(_spec(round_window=WINDOW), pair.root)

    assert result.valid, result.reason
    for t in WINDOW:
        assert result.outcome_metrics[f"round_{t}_accuracy"] == (
            pair.ref_round_metrics[t]["accuracy"]
        )
        assert result.outcome_metrics[f"round_{t}_loss"] == (
            pair.ref_round_metrics[t]["loss"]
        )


def test_windowed_inject_reproduces_degradation(pair):
    result = apply_intervention(
        _spec(target_run_id="ref", source_run_id="fail", mode="inject",
              round_window=WINDOW),
        pair.root,
    )

    assert result.valid, result.reason
    # injecting the failure window into the reference replays it exactly
    # into the failure run's outcome
    assert result.outcome_metrics["accuracy"] == pair.fail_final["accuracy"]
    assert result.outcome_metrics["loss"] == pair.fail_final["loss"]
    assert result.outcome_metrics["loss"] > pair.ref_final["loss"]


def test_windowed_sham_produces_zero_metric_deviation(pair):
    result = apply_intervention(
        _spec(mode="sham", round_window=WINDOW), pair.root
    )

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


# --- window validation failure paths (no partial windows) ------------------


def test_invalid_round_inside_window_rejects_whole_spec(pair):
    """Round t1 alone would be valid; a rogue recording at t2 must reject ALL."""
    name = _craft_run(pair, "ref", "rogue_window")
    recorder = Recorder(pair.root, name)
    state = recorder.load(WINDOW[1], "compression")
    recorder.record(
        WINDOW[1],
        "compression",
        [entry.model_copy(update={"round_id": 99}) for entry in state],
    )

    result = apply_intervention(
        _spec(source_run_id=name, round_window=WINDOW), pair.root
    )

    assert not result.valid
    assert "source_round_mismatch" in result.reason
    # the reason names the rejecting window round
    assert result.reason.endswith(f":{WINDOW[1]}")


@pytest.mark.parametrize("role", ["target", "source"])
def test_window_must_be_inside_recorded_rounds(pair, role):
    """window ∩ recorded rounds: a recording that stops before t2 rejects."""
    name = _craft_run(pair, "ref" if role == "source" else "fail", f"gap_{role}")
    shutil.rmtree(pair.root / "runs" / name / f"round_{WINDOW[1]}")
    overrides = {f"{role}_run_id": name}

    result = apply_intervention(
        _spec(round_window=WINDOW, **overrides), pair.root
    )

    assert not result.valid
    assert f"{role}_boundary_missing" in result.reason
    assert result.reason.endswith(f":{WINDOW[1]}")


@pytest.mark.parametrize(
    "window", [(-1, 2), (0, ROUNDS), (3, 2)], ids=["negative", "past_end", "reversed"]
)
def test_out_of_range_window_is_invalid(pair, window):
    result = apply_intervention(_spec(round_window=window), pair.root)
    assert not result.valid
    assert "invalid_round_window" in result.reason


# --- CLI -------------------------------------------------------------------


def test_cli_round_window_runs_and_writes_json(pair, tmp_path):
    json_path = tmp_path / "result.json"
    exit_code = cli_main(
        [
            "--runs-root", str(pair.root),
            "--target-run", "fail",
            "--source-run", "ref",
            "--round-window", f"{WINDOW[0]}:{WINDOW[1]}",
            "--stage", "compression",
            "--mode", "restore",
            "--json", str(json_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["spec"]["round_window"] == list(WINDOW)
    assert payload["outcome_metrics"]["loss"] == pair.ref_final["loss"]
    assert f"round_{WINDOW[0]}_loss" in payload["outcome_metrics"]
    assert f"round_{WINDOW[1]}_loss" in payload["outcome_metrics"]


def test_cli_round_and_round_window_are_mutually_exclusive(pair):
    with pytest.raises(SystemExit):
        cli_main(
            [
                "--runs-root", str(pair.root),
                "--target-run", "fail",
                "--source-run", "ref",
                "--round", "2",
                "--round-window", "2:3",
                "--stage", "compression",
                "--mode", "restore",
            ]
        )
