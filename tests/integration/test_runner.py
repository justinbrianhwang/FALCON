"""Integration tests for the round loop (Task T2): 5 clients, 3 rounds."""
import hashlib

import numpy as np
import pytest

from falcon.pipeline.runner import run
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    LocalConfig,
    RunConfig,
    SelectionConfig,
)


class StubRng:
    """Minimal inline stand-in for falcon.replay.rng.Rng (CONTRACTS §3)."""

    def __init__(self, seed: int):
        self._seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            self._streams[name] = np.random.default_rng(np.random.SeedSequence([self._seed, key]))
        return self._streams[name]


@pytest.fixture
def cfg() -> RunConfig:
    return RunConfig(
        run_id="test_synthetic",
        seed=42,
        rounds=3,
        dataset=DatasetConfig(
            num_clients=5, num_features=10, num_classes=2, samples_per_client=80
        ),
        selection=SelectionConfig(clients_per_round=3),
        local=LocalConfig(lr=0.5, local_steps=8, batch_size=16),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
    )


def test_loss_decreases_over_rounds(cfg):
    outcomes = run(cfg, rng=StubRng(cfg.seed))
    assert len(outcomes) == 3
    assert [o.round_id for o in outcomes] == [0, 1, 2]
    losses = [o.metrics["loss"] for o in outcomes]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert outcomes[-1].metrics["accuracy"] >= outcomes[0].metrics["accuracy"]


def test_two_runs_identical(cfg):
    a = run(cfg, rng=StubRng(cfg.seed))
    b = run(cfg, rng=StubRng(cfg.seed))
    assert [o.model_hash for o in a] == [o.model_hash for o in b]
    assert [o.metrics for o in a] == [o.metrics for o in b]
    assert [o.per_class for o in a] == [o.per_class for o in b]


def test_recorder_called_at_every_boundary(cfg):
    calls: list[tuple[int, str, object]] = []

    class FakeRecorder:
        def record(self, round_id, stage, state):
            calls.append((round_id, stage, state))

    run(cfg, recorder=FakeRecorder(), rng=StubRng(cfg.seed))
    stages_by_round = {}
    for round_id, stage, _state in calls:
        stages_by_round.setdefault(round_id, []).append(stage)
    assert sorted(stages_by_round) == [0, 1, 2]
    for round_id in (0, 1, 2):
        stages = stages_by_round[round_id]
        assert stages.count("selection") == 1
        assert stages.count("local") == cfg.selection.clients_per_round
        assert stages.count("compression") == cfg.selection.clients_per_round
        assert stages.count("aggregation") == 1
        assert stages.count("evaluation") == 1
