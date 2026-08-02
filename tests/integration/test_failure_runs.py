"""Integration tests for the four T4 failure cases (docs/tasks/T4).

Each case file in ``configs/cases/`` is paired with its own reference: the
same YAML with ``failure`` stripped. With the same seed, runs must agree on
every stage hash before the failure window, and the first divergent stage
must be exactly the injected stage. The failure run's primary metric must be
measurably worse — and since T11 hardened the task (class_separation 0.4,
label_noise 0.1, no more saturated accuracy) EVERY case must show a
measurable ACCURACY gap, not just a loss gap (Plan §17.3 forbids
"no measurable failure" severities). Chosen severities and the exact
deterministic gaps measured on the committed configs:

- selection / minority_exclusion: p=1.0 on rounds 2-9 drops the two
  minority-heavy clients (each holds ~45% of class 1 vs 10% global share).
  Primary metric: minority-class eval accuracy (Plan §10.2 S1).
  Measured: minority-class accuracy gap +0.165, overall accuracy gap +0.078.
- local / lr_misconfig: four clients get a sign-flipped lr (multiplier -1.0,
  Plan §10.2 L1 "sign error" sanity case — a merely larger/smaller rate does
  not measurably degrade even on the hardened task). Primary: accuracy/loss.
  Measured: accuracy gap +0.142, loss gap +0.115.
- compression / aggressive_topk: k_ratio 0.05 keeps ceil(0.05*42) = 3 of 42
  update coordinates on rounds 2-9. Primary: accuracy/loss.
  Measured: accuracy gap +0.126, loss gap +0.096.
- aggregation / wrong_sample_weights: "corrupted" mode (log-uniform factors
  in [0.1, 10]) on rounds 2-9, under heterogeneity 2.0 so reweighting bites
  (uniform weights are a no-op here: all clients hold 100 samples). T11
  calibration: clients_per_round 5 -> 2, because with 5 clients the
  corrupted weights re-normalize away (accuracy gap was +0.002, not
  measurable). Primary: accuracy/loss.
  Measured: accuracy gap +0.070, loss gap +0.041.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    FailureSpecification,
    LocalConfig,
    RunConfig,
    SelectionConfig,
)

CASES_DIR = Path(__file__).resolve().parents[2] / "configs" / "cases"
STAGE_ORDER = ("selection", "local", "compression", "aggregation", "evaluation")

CASES = [
    ("synthetic_selection_failure.yaml", "selection"),
    ("synthetic_local_failure.yaml", "local"),
    ("synthetic_compression_failure.yaml", "compression"),
    ("synthetic_aggregation_failure.yaml", "aggregation"),
]


def _load_pair(filename: str) -> tuple[RunConfig, RunConfig]:
    payload = yaml.safe_load((CASES_DIR / filename).read_text(encoding="utf-8"))
    failure_cfg = RunConfig(**payload)
    reference_payload = copy.deepcopy(payload)
    reference_payload["failure"] = None
    reference_payload["run_id"] = payload["run_id"].replace("_failure", "_reference")
    return RunConfig(**reference_payload), failure_cfg


def _run_recorded(cfg: RunConfig, root: Path, run_id: str):
    recorder = Recorder(root, run_id)
    outcomes = run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder.stage_hashes(), outcomes


def _first_divergence(reference_hashes, failure_hashes, rounds):
    """First (round_id, stage) whose recorded hash differs, in pipeline order."""
    for round_id in range(rounds):
        for stage in STAGE_ORDER:
            key = (round_id, stage)
            if reference_hashes[key] != failure_hashes[key]:
                return key
    return None


@pytest.mark.parametrize("filename, injected_stage", CASES, ids=[c[0] for c in CASES])
def test_pre_window_hashes_identical_and_first_divergence_at_injected_stage(
    tmp_path, filename, injected_stage
):
    reference_cfg, failure_cfg = _load_pair(filename)
    assert failure_cfg.failure is not None
    assert failure_cfg.failure.stage == injected_stage

    reference_hashes, _ = _run_recorded(reference_cfg, tmp_path, "reference")
    failure_hashes, _ = _run_recorded(failure_cfg, tmp_path, "failure")

    active_start, active_end = failure_cfg.failure.active_rounds
    assert active_start > 0  # every case keeps a clean pre-failure window

    # before the window every recorded stage boundary is identical
    for round_id in range(active_start):
        for stage in STAGE_ORDER:
            assert reference_hashes[(round_id, stage)] == failure_hashes[(round_id, stage)], (
                f"unexpected divergence at round {round_id} stage {stage}"
            )

    # the first divergence happens inside the window at the injected stage
    first = _first_divergence(reference_hashes, failure_hashes, failure_cfg.rounds)
    assert first is not None, "failure run never diverged from the reference"
    assert first[1] == injected_stage
    assert active_start <= first[0] <= active_end


@pytest.mark.parametrize("filename, injected_stage", CASES, ids=[c[0] for c in CASES])
def test_failure_run_metric_measurably_worse(tmp_path, filename, injected_stage):
    reference_cfg, failure_cfg = _load_pair(filename)
    _, reference = _run_recorded(reference_cfg, tmp_path, "reference")
    _, failure = _run_recorded(failure_cfg, tmp_path, "failure")
    ref_final, fail_final = reference[-1], failure[-1]

    if injected_stage == "selection":
        # minority recall is the primary metric for S1 (Plan §10.2);
        # measured on the committed config: +0.165 minority, +0.078 overall
        gap = ref_final.per_class["1"]["accuracy"] - fail_final.per_class["1"]["accuracy"]
        assert gap >= 0.10, f"minority-class accuracy gap too small: {gap}"
        acc_gap = ref_final.metrics["accuracy"] - fail_final.metrics["accuracy"]
        assert acc_gap >= 0.05, f"accuracy gap too small: {acc_gap}"
    elif injected_stage == "local":
        # measured on the committed config: +0.142 accuracy, +0.115 loss
        gap = ref_final.metrics["accuracy"] - fail_final.metrics["accuracy"]
        assert gap >= 0.10, f"accuracy gap too small: {gap}"
        assert fail_final.metrics["loss"] > ref_final.metrics["loss"] + 0.05
    elif injected_stage == "compression":
        # measured on the committed config: +0.126 accuracy, +0.096 loss
        gap = ref_final.metrics["accuracy"] - fail_final.metrics["accuracy"]
        assert gap >= 0.10, f"accuracy gap too small: {gap}"
        assert fail_final.metrics["loss"] > ref_final.metrics["loss"] + 0.05
    else:  # aggregation
        # measured on the committed config: +0.070 accuracy, +0.041 loss
        acc_gap = ref_final.metrics["accuracy"] - fail_final.metrics["accuracy"]
        assert acc_gap >= 0.03, f"accuracy gap too small: {acc_gap}"
        gap = fail_final.metrics["loss"] - ref_final.metrics["loss"]
        assert gap >= 0.02, f"loss gap too small: {gap}"


def test_reference_run_of_each_case_is_deterministic(tmp_path):
    """The stripped reference (failure=None) replays byte-identically."""
    for filename, _ in CASES:
        reference_cfg, _ = _load_pair(filename)
        hashes_a, _ = _run_recorded(reference_cfg, tmp_path, f"det-a-{filename}")
        hashes_b, _ = _run_recorded(reference_cfg, tmp_path, f"det-b-{filename}")
        assert hashes_a == hashes_b


def test_exclusion_undersubscribed_round_completes_and_is_recorded(tmp_path):
    """T4-F finding 7: exclusion may shrink the pool below clients_per_round.

    8 of 10 clients are minority-heavy; p=1.0 exclusion on rounds 1-2 leaves a
    2-client pool against clients_per_round=5. The run must complete (no
    sampling crash) and the recorded rounds must show the shortfall.
    """
    cfg = RunConfig(
        run_id="undersubscribed",
        seed=42,
        rounds=3,
        dataset=DatasetConfig(
            num_clients=10,
            num_features=10,
            num_classes=2,
            samples_per_client=100,
            minority_class=1,
            minority_client_fraction=0.8,
        ),
        selection=SelectionConfig(clients_per_round=5),
        local=LocalConfig(lr=0.1, local_steps=3, batch_size=32),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
        failure=FailureSpecification(
            stage="selection",
            type="minority_exclusion",
            active_rounds=(1, 2),
            severity=1,
            parameters={"target_class": 1, "exclusion_probability": 1.0},
        ),
    )
    recorder = Recorder(tmp_path, "undersubscribed")
    outcomes = run(cfg, recorder=recorder, rng=Rng(cfg.seed))

    assert len(outcomes) == cfg.rounds  # the run completed
    assert all(np.isfinite(o.metrics["loss"]) for o in outcomes)
    # round 0 is outside the window: full 5-client selection
    assert len(recorder.load(0, "selection").selected_ids) == 5
    # active rounds: 8 of 10 clients excluded -> undersubscribed 2-client rounds
    for round_id in (1, 2):
        selection = recorder.load(round_id, "selection")
        assert len(selection.candidate_ids) == 2
        assert selection.selected_ids == selection.candidate_ids
        assert len(selection.selected_ids) < cfg.selection.clients_per_round
        assert selection.sampling_probs == {
            cid: pytest.approx(1.0) for cid in selection.candidate_ids
        }
        # downstream stages ran with exactly the undersubscribed clients
        local_states = recorder.load(round_id, "local")
        assert [s.client_id for s in local_states] == selection.selected_ids
