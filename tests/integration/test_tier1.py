"""Integration tests for Tier 1 (Task T18): real datasets + SmallCNN (torch).

- MNIST tiny config (``configs/cases/mnist_reference.yaml``): a clean
  duplicate replay is bit-identical across ALL recorded stage hashes
  (E0-style, CONTRACTS §5); the loss decreases; per-client stages are
  recorded once per stage as LISTS of states (CONTRACTS §1).
- T4 failure aggressive_topk on MNIST: measurable accuracy gap, and the
  first divergence lands exactly on the compression stage — the Tier-1
  analog of tests/integration/test_failure_runs.py.
- torch must stay out of falcon/schema and the synthetic pipeline path;
  asserted in a subprocess because this module itself imports torch through
  the pipeline (so ``sys.modules`` here is already tainted).
- CIFAR-10 determinism smoke is marked ``slow`` (runnable manually).

Data-dependent tests skip cleanly when the processed pickle is absent
(run ``python scripts/prepare_data.py --datasets mnist,cifar10`` first).
"""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip(
    "torch", reason="Tier-1 tests need torch (use the falcon conda env python)"
)

from falcon.data_paths import processed_path
from falcon.pipeline.models import build_model, flatten
from falcon.pipeline.runner import run
from falcon.pipeline.synthetic_data import ClientData
from falcon.pipeline.torch_local import local_train
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    DatasetConfig,
    FailureSpecification,
    LocalConfig,
    RunConfig,
)
from falcon.schema.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "configs" / "cases"
STAGE_ORDER = ("selection", "local", "compression", "aggregation", "evaluation")


def _load_case(filename: str) -> RunConfig:
    payload = yaml.safe_load((CASES_DIR / filename).read_text(encoding="utf-8"))
    return RunConfig(**payload)


def _require(dataset: str) -> None:
    if not processed_path(dataset).exists():
        pytest.skip(
            f"processed {dataset}.pkl not found — run "
            f"`python scripts/prepare_data.py --datasets {dataset}` first"
        )


def _first_divergence(reference_hashes, failure_hashes, rounds):
    """First (round_id, stage) whose recorded hash differs, in pipeline order."""
    for round_id in range(rounds):
        for stage in STAGE_ORDER:
            key = (round_id, stage)
            if reference_hashes[key] != failure_hashes[key]:
                return key
    return None


def test_torch_local_supports_negative_lr_sign_error():
    dataset = DatasetConfig(name="mnist", num_clients=1, num_classes=10)
    model_cfg = ModelConfig(name="small_cnn")
    initial = flatten(build_model(model_cfg, dataset, Rng(7)))
    data = ClientData(
        x=np.zeros((4, 1, 28, 28), dtype=np.float32),
        y=np.array([0, 1, 2, 3], dtype=np.int64),
    )

    def train(lr):
        return local_train(
            initial,
            "client_0",
            data,
            0,
            LocalConfig(lr=lr, local_steps=1, batch_size=4),
            Rng(11),
            model_cfg=model_cfg,
            dataset_cfg=dataset,
        )

    positive = train(0.1)
    negative = train(-0.1)
    np.testing.assert_allclose(negative.update, -positive.update, rtol=1e-4, atol=1e-7)


@pytest.fixture(scope="module")
def mnist_reference(tmp_path_factory):
    """One recorded run of the committed MNIST reference case, shared by tests."""
    _require("mnist")
    cfg = _load_case("mnist_reference.yaml")
    recorder = Recorder(tmp_path_factory.mktemp("tier1_mnist"), "mnist-reference-a")
    outcomes = run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return cfg, recorder, outcomes


def test_mnist_loss_decreases(mnist_reference):
    """Clean run learns: measured accuracy 0.262 -> 0.763, loss 2.252 -> 0.725."""
    _, _, outcomes = mnist_reference
    assert len(outcomes) == 5
    assert [o.round_id for o in outcomes] == list(range(5))
    losses = [o.metrics["loss"] for o in outcomes]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    accuracy = outcomes[-1].metrics["accuracy"]
    assert accuracy > 0.5, f"final accuracy too low: {accuracy}"


def test_mnist_clean_duplicate_replay_bit_identical(mnist_reference, tmp_path):
    """Same config + seed => identical content_hash at every stage boundary."""
    cfg, recorder_a, outcomes_a = mnist_reference
    recorder_b = Recorder(tmp_path, "mnist-reference-b")
    outcomes_b = run(cfg, recorder=recorder_b, rng=Rng(cfg.seed))

    hashes_a, hashes_b = recorder_a.stage_hashes(), recorder_b.stage_hashes()
    expected = {
        (round_id, stage) for round_id in range(cfg.rounds) for stage in STAGE_ORDER
    }
    assert set(hashes_a) == expected
    assert hashes_a == hashes_b
    assert [o.model_hash for o in outcomes_a] == [o.model_hash for o in outcomes_b]
    assert [o.metrics for o in outcomes_a] == [o.metrics for o in outcomes_b]

    # recorded arrays come back bit-identical across the duplicate runs
    for round_id in range(cfg.rounds):
        for stage in ("local", "compression"):
            states_a = recorder_a.load(round_id, stage)
            states_b = recorder_b.load(round_id, stage)
            for state_a, state_b in zip(states_a, states_b):
                assert state_a.update.dtype == state_b.update.dtype == np.float32
                assert np.array_equal(state_a.update, state_b.update)


def test_mnist_per_client_stages_recorded_as_lists(mnist_reference):
    cfg, recorder, _ = mnist_reference
    for round_id in range(cfg.rounds):
        selection = recorder.load(round_id, "selection")
        local_states = recorder.load(round_id, "local")
        compressed = recorder.load(round_id, "compression")
        assert isinstance(local_states, list) and isinstance(compressed, list)
        assert [s.client_id for s in local_states] == selection.selected_ids
        assert [c.client_id for c in compressed] == selection.selected_ids
        # Tier-1 flat arrays are torch-native float32 (CONTRACTS-adjacent note)
        assert all(s.update.dtype == np.float32 for s in local_states)
        assert all(c.update.dtype == np.float32 for c in compressed)
        aggregation = recorder.load(round_id, "aggregation")
        assert aggregation.aggregate.dtype == np.float32
        assert aggregation.accepted_ids == selection.selected_ids


def test_mnist_aggressive_topk_failure(mnist_reference, tmp_path):
    """T4 compression failure on MNIST: first divergence at ``compression``,
    measurable accuracy/loss gap (Plan §17.3).

    k_ratio 0.01 keeps ceil(0.01 * 421642) = 4217 of the SmallCNN's 421642
    update coordinates on rounds 2-4. Measured on the committed config:
    accuracy gap +0.059, loss gap +0.224.
    """
    cfg, recorder_a, outcomes_a = mnist_reference
    payload = yaml.safe_load((CASES_DIR / "mnist_reference.yaml").read_text(encoding="utf-8"))
    failure_payload = copy.deepcopy(payload)
    failure_payload["run_id"] = "mnist_compression_failure"
    failure_payload["failure"] = {
        "stage": "compression",
        "type": "aggressive_topk",
        "active_rounds": [2, 4],
        "severity": 2,
        "parameters": {"k_ratio": 0.01},
    }
    failure_cfg = RunConfig(**failure_payload)
    assert failure_cfg.failure is not None

    recorder_b = Recorder(tmp_path, "mnist-compression-failure")
    outcomes_b = run(failure_cfg, recorder=recorder_b, rng=Rng(failure_cfg.seed))

    hashes_a, hashes_b = recorder_a.stage_hashes(), recorder_b.stage_hashes()
    # rounds 0-1 are outside the failure window: every boundary identical
    for round_id in range(2):
        for stage in STAGE_ORDER:
            assert hashes_a[(round_id, stage)] == hashes_b[(round_id, stage)]
    first = _first_divergence(hashes_a, hashes_b, cfg.rounds)
    assert first is not None, "failure run never diverged from the reference"
    assert first[1] == "compression"
    assert 2 <= first[0] <= 4

    ref_final, fail_final = outcomes_a[-1], outcomes_b[-1]
    acc_gap = ref_final.metrics["accuracy"] - fail_final.metrics["accuracy"]
    assert acc_gap >= 0.03, f"accuracy gap too small: {acc_gap}"
    loss_gap = fail_final.metrics["loss"] - ref_final.metrics["loss"]
    assert loss_gap >= 0.10, f"loss gap too small: {loss_gap}"


def test_torch_absent_from_schema_and_synthetic_path():
    """falcon/schema and a full synthetic run must not import torch (T18 rule)."""
    code = (
        "import sys\n"
        "import falcon.schema\n"
        "assert 'torch' not in sys.modules, 'falcon.schema pulled in torch'\n"
        "from falcon.pipeline.runner import run\n"
        "assert 'torch' not in sys.modules, 'pipeline.runner pulled in torch'\n"
        "from falcon.schema import (AggregationConfig, CompressionConfig,\n"
        "    DatasetConfig, LocalConfig, RunConfig, SelectionConfig)\n"
        "cfg = RunConfig(\n"
        "    run_id='no-torch', seed=42, rounds=2,\n"
        "    dataset=DatasetConfig(num_clients=4, num_features=6, num_classes=2,\n"
        "                          samples_per_client=40),\n"
        "    selection=SelectionConfig(clients_per_round=2),\n"
        "    local=LocalConfig(lr=0.1, local_steps=2, batch_size=16),\n"
        "    compression=CompressionConfig(kind='identity'),\n"
        "    aggregation=AggregationConfig(rule='weighted_mean'),\n"
        ")\n"
        "run(cfg)\n"
        "assert 'torch' not in sys.modules, 'synthetic run pulled in torch'\n"
        "print('torch-free ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert "torch-free ok" in completed.stdout


def test_flatten_load_flat_round_trip():
    """The flat-vector bridge: float32, registration order, exact round-trip."""
    from falcon.pipeline.models import SmallCNN, flatten, load_flat

    model = SmallCNN(1, 28, 10)
    vec = flatten(model)
    assert vec.dtype == np.float32 and vec.ndim == 1
    assert vec.shape[0] == sum(p.numel() for p in model.parameters())

    other = SmallCNN(1, 28, 10)
    load_flat(other, vec)
    assert np.array_equal(flatten(other), vec)
    # 32x32x3 input adapts the fc width (421642 params for 28x28x1 @ 10 classes)
    cifar_model = SmallCNN(3, 32, 10)
    assert flatten(cifar_model).shape[0] != vec.shape[0]
    with pytest.raises(ValueError):
        load_flat(cifar_model, vec)


def test_mnist_minority_concentration_honored():
    """minority_class/minority_client_fraction honored like synthetic (T18):
    the designated 20% of clients hold ALL class-3 samples and are exactly the
    minority-heavy set of the T4 targeting rule; the partition is reproducible
    from DatasetConfig.seed alone."""
    _require("mnist")
    from falcon.failures.targeting import minority_heavy_clients
    from falcon.pipeline.real_data import load_partition
    from falcon.schema import DatasetConfig

    cfg = DatasetConfig(
        name="mnist",
        num_clients=10,
        num_classes=10,
        dirichlet_alpha=0.5,
        minority_class=3,
        minority_client_fraction=0.2,
    )
    partition = load_partition(cfg)
    holders = {cid for cid, data in partition.items() if np.any(data.y == 3)}
    assert len(holders) == 2  # max(1, round(10 * 0.2)) designated clients
    assert holders == set(minority_heavy_clients(partition, 3))
    # reproducible: same DatasetConfig => identical partition, run seed irrelevant
    again = load_partition(cfg)
    for cid in partition:
        assert np.array_equal(partition[cid].y, again[cid].y)


@pytest.mark.slow
def test_cifar10_determinism_smoke(tmp_path):
    """CIFAR-10 Tier-1 smoke: two recorded 2-round runs match on every hash."""
    _require("cifar10")
    cfg = _load_case("cifar10_reference.yaml")
    cfg = cfg.model_copy(update={"rounds": 2})
    recorder_a = Recorder(tmp_path, "cifar-det-a")
    recorder_b = Recorder(tmp_path, "cifar-det-b")
    run(cfg, recorder=recorder_a, rng=Rng(cfg.seed))
    run(cfg, recorder=recorder_b, rng=Rng(cfg.seed))
    assert recorder_a.stage_hashes() == recorder_b.stage_hashes()
