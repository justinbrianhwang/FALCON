"""Integration tests for the round loop (Task T2): 5 clients, 3 rounds.

Recording protocol (CONTRACTS §1/§4): per-client stages ("local",
"compression") are recorded ONCE per stage as a list of states; every other
stage is recorded as a single state.
"""
import numpy as np
import pytest

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    ClientLocalState,
    CompressionConfig,
    CompressionState,
    DatasetConfig,
    LocalConfig,
    RunConfig,
    SelectionConfig,
)


@pytest.fixture
def cfg() -> RunConfig:
    return RunConfig(
        run_id="test_synthetic",
        seed=42,
        rounds=3,
        # T11-hardened task (class_separation 0.4, label_noise 0.1): accuracy
        # must not saturate at ~1.0 the way the separable task did.
        dataset=DatasetConfig(
            num_clients=5,
            num_features=10,
            num_classes=2,
            samples_per_client=80,
            class_separation=0.4,
            label_noise=0.1,
        ),
        selection=SelectionConfig(clients_per_round=3),
        local=LocalConfig(lr=0.5, local_steps=8, batch_size=16),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
    )


def test_loss_decreases_over_rounds(cfg):
    outcomes = run(cfg, rng=Rng(cfg.seed))
    assert len(outcomes) == 3
    assert [o.round_id for o in outcomes] == [0, 1, 2]
    losses = [o.metrics["loss"] for o in outcomes]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    # learning happened but the hardened task is not saturated (T11):
    # measured accuracy on this fixture is ~0.80
    accuracy = outcomes[-1].metrics["accuracy"]
    assert 0.6 < accuracy < 0.95


def test_two_runs_identical(cfg):
    """End-to-end determinism with the production Rng (CONTRACTS §5)."""
    a = run(cfg, rng=Rng(cfg.seed))
    b = run(cfg, rng=Rng(cfg.seed))
    assert [o.model_hash for o in a] == [o.model_hash for o in b]
    assert [o.metrics for o in a] == [o.metrics for o in b]
    assert [o.per_class for o in a] == [o.per_class for o in b]


def test_median_rule_run_deterministic_and_learns(cfg):
    """T21: synthetic reference run with rule ``median`` (E5 prerequisite).

    Completes all rounds, two runs are bit-identical (model hashes and
    metrics), and clean-run accuracy stays in a sane band — measured ~0.79
    on this fixture, comparable to the weighted_mean reference.
    """
    median_cfg = cfg.model_copy(update={"aggregation": AggregationConfig(rule="median")})
    a = run(median_cfg, rng=Rng(median_cfg.seed))
    b = run(median_cfg, rng=Rng(median_cfg.seed))
    assert len(a) == median_cfg.rounds
    assert [o.model_hash for o in a] == [o.model_hash for o in b]
    assert [o.metrics for o in a] == [o.metrics for o in b]
    losses = [o.metrics["loss"] for o in a]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    accuracy = a[-1].metrics["accuracy"]
    assert 0.6 < accuracy < 0.95


def test_default_rng_matches_injected_rng(cfg):
    """run(cfg) must default to Rng(cfg.seed) (blessed signature, CONTRACTS §1)."""
    assert [o.model_hash for o in run(cfg)] == [
        o.model_hash for o in run(cfg, rng=Rng(cfg.seed))
    ]


def test_recorder_called_once_per_stage_with_lists(cfg):
    calls: list[tuple[int, str, object]] = []

    class FakeRecorder:
        def record(self, round_id, stage, state):
            calls.append((round_id, stage, state))

    run(cfg, recorder=FakeRecorder(), rng=Rng(cfg.seed))

    stages_by_round: dict[int, list[str]] = {}
    for round_id, stage, _state in calls:
        stages_by_round.setdefault(round_id, []).append(stage)
    assert sorted(stages_by_round) == [0, 1, 2]
    for round_id in (0, 1, 2):
        # exactly one record call per stage per round — per-client stages too
        assert sorted(stages_by_round[round_id]) == [
            "aggregation",
            "compression",
            "evaluation",
            "local",
            "selection",
        ]

    for round_id, stage, state in calls:
        if stage not in ("local", "compression"):
            continue
        selection = next(
            s for r, t, s in calls if r == round_id and t == "selection"
        )
        expected_type = ClientLocalState if stage == "local" else CompressionState
        assert isinstance(state, list)
        assert all(isinstance(s, expected_type) for s in state)
        assert [s.client_id for s in state] == selection.selected_ids


class _TapRecorder:
    """Wraps a real Recorder, keeping the in-memory states it was given."""

    def __init__(self, inner: Recorder):
        self._inner = inner
        self.calls: list[tuple[int, str, object]] = []

    def record(self, round_id, stage, state):
        self.calls.append((round_id, stage, state))
        self._inner.record(round_id, stage, state)


def test_real_recorder_per_client_stages_round_trip(cfg, tmp_path):
    recorder = Recorder(tmp_path, "round_trip")
    tap = _TapRecorder(recorder)
    outcomes = run(cfg, recorder=tap, rng=Rng(cfg.seed))

    for round_id in range(cfg.rounds):
        selection = recorder.load(round_id, "selection")
        local_states = recorder.load(round_id, "local")
        compressed = recorder.load(round_id, "compression")

        # per-client stages load back as lists aligned with the selection
        assert isinstance(local_states, list) and isinstance(compressed, list)
        assert [s.client_id for s in local_states] == selection.selected_ids
        assert [c.client_id for c in compressed] == selection.selected_ids

        # arrays come back bit-identical to the in-memory states that were recorded
        in_memory = {s.client_id: s for r, t, lst in tap.calls if r == round_id and t == "local" for s in lst}
        for loaded in local_states:
            original = in_memory[loaded.client_id]
            assert np.array_equal(loaded.update, original.update)
            assert loaded.update.dtype == np.float64
            assert loaded.loss_history == original.loss_history
            assert loaded.rng_state == original.rng_state
        for loaded, original in zip(local_states, compressed):
            assert np.array_equal(loaded.update, original.update)

        aggregation = recorder.load(round_id, "aggregation")
        assert aggregation.accepted_ids == selection.selected_ids
        evaluation = recorder.load(round_id, "evaluation")
        assert evaluation.metrics == outcomes[round_id].metrics


def test_stage_hashes_equal_across_two_clean_runs(cfg, tmp_path):
    rec_a = Recorder(tmp_path, "clean-a")
    rec_b = Recorder(tmp_path, "clean-b")
    run(cfg, recorder=rec_a, rng=Rng(cfg.seed))
    run(cfg, recorder=rec_b, rng=Rng(cfg.seed))

    hashes_a = rec_a.stage_hashes()
    hashes_b = rec_b.stage_hashes()
    expected_boundaries = {
        (round_id, stage)
        for round_id in range(cfg.rounds)
        for stage in ("selection", "local", "compression", "aggregation", "evaluation")
    }
    assert set(hashes_a) == expected_boundaries
    assert hashes_a == hashes_b

    # recorded arrays are bit-identical across clean runs
    for round_id in range(cfg.rounds):
        for stage in ("local", "compression"):
            states_a = rec_a.load(round_id, stage)
            states_b = rec_b.load(round_id, stage)
            assert [s.client_id for s in states_a] == [s.client_id for s in states_b]
            for state_a, state_b in zip(states_a, states_b):
                assert np.array_equal(state_a.update, state_b.update)
