"""Unit tests for the five stage functions (Task T2, CONTRACTS §1)."""
import copy
import hashlib
import json

import numpy as np
import pytest

from falcon.pipeline.stages import (
    aggregate,
    compress,
    evaluate,
    init_params,
    local_train,
    select_clients,
)
from falcon.pipeline.synthetic_data import ClientData, make_partition
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    LocalConfig,
    SelectionConfig,
)


class StubRng:
    """Minimal inline stand-in for falcon.replay.rng.Rng (CONTRACTS §3).

    Named, independent streams keyed by a stable hash of the stream name
    (order-independent), derived from a single root seed.
    """

    def __init__(self, seed: int):
        self._seed = seed
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            self._streams[name] = np.random.default_rng(np.random.SeedSequence([self._seed, key]))
        return self._streams[name]


class SpyRng:
    """Wraps the production Rng, recording every requested stream name.

    If ``allowed`` is given, requesting a stream outside that set fails the
    test immediately — stages must touch only their contracted streams.
    """

    def __init__(self, seed: int, allowed: set[str] | None = None):
        self._rng = Rng(seed)
        self._allowed = allowed
        self.requested: list[str] = []

    def stream(self, name: str) -> np.random.Generator:
        self.requested.append(name)
        if self._allowed is not None and name not in self._allowed:
            raise AssertionError(f"unexpected stream requested: {name!r}")
        return self._rng.stream(name)


@pytest.fixture
def dataset_cfg() -> DatasetConfig:
    return DatasetConfig(num_clients=3, num_features=5, num_classes=2, samples_per_client=40)


@pytest.fixture
def client_data(dataset_cfg) -> ClientData:
    return make_partition(dataset_cfg)["client_0"]


@pytest.fixture
def params() -> np.ndarray:
    return init_params(5, 2, StubRng(0))


def test_select_clients_deterministic_and_uniform():
    pool = [f"client_{i}" for i in range(10)]
    cfg = SelectionConfig(clients_per_round=4)
    a = select_clients(pool, 0, cfg, StubRng(7))
    b = select_clients(pool, 0, cfg, StubRng(7))
    assert a.selected_ids == b.selected_ids
    assert len(a.selected_ids) == 4
    assert len(set(a.selected_ids)) == 4  # without replacement
    assert set(a.selected_ids) <= set(pool)
    assert a.sampling_probs == {cid: pytest.approx(0.4) for cid in pool}
    assert a.round_id == 0 and a.candidate_ids == pool


def test_local_train_deterministic(client_data, params):
    cfg = LocalConfig(lr=0.5, local_steps=6, batch_size=8)
    a = local_train(params, "client_0", client_data, 0, cfg, StubRng(11))
    b = local_train(params, "client_0", client_data, 0, cfg, StubRng(11))
    assert np.array_equal(a.update, b.update)
    assert a.loss_history == b.loss_history
    assert len(a.loss_history) == cfg.local_steps
    assert all(np.isfinite(a.loss_history))
    assert a.num_steps == cfg.local_steps
    assert a.num_examples == client_data.x.shape[0]
    assert a.update.shape == params.shape
    expected_hash = hashlib.sha256(params.tobytes()).hexdigest()
    assert a.base_model_hash == expected_hash


def test_local_train_update_is_delta(client_data, params):
    cfg = LocalConfig(lr=0.0, local_steps=3, batch_size=8)
    state = local_train(params, "client_0", client_data, 0, cfg, StubRng(3))
    # lr = 0 => trained == global => delta is exactly zero
    assert np.array_equal(state.update, np.zeros_like(params))


def test_compress_identity_round_trips_exactly(client_data, params):
    local_cfg = LocalConfig(lr=0.5, local_steps=2, batch_size=8)
    local = local_train(params, "client_0", client_data, 0, local_cfg, StubRng(5))
    comp = compress(local, CompressionConfig(kind="identity"), StubRng(5))
    assert np.array_equal(comp.update, local.update)
    assert comp.update.dtype == np.float64
    assert comp.update is not local.update  # a copy, not an alias
    assert comp.uncompressed_hash == hashlib.sha256(local.update.tobytes()).hexdigest()
    assert comp.bytes_transmitted == local.update.nbytes
    assert comp.client_id == local.client_id and comp.round_id == local.round_id


def test_compress_quantization_not_implemented(client_data, params):
    local_cfg = LocalConfig(lr=0.5, local_steps=1, batch_size=8)
    local = local_train(params, "client_0", client_data, 0, local_cfg, StubRng(5))
    with pytest.raises(NotImplementedError):
        compress(local, CompressionConfig(kind="quantization"), StubRng(5))


def _compressed(client_id: str, update: np.ndarray, round_id: int = 0):
    local_cfg = LocalConfig(lr=0.0, local_steps=1, batch_size=1)
    data = ClientData(
        x=np.zeros((1, update.shape[0] // 2 - 1), dtype=np.float64),
        y=np.zeros(1, dtype=np.int64),
    )
    base = np.zeros_like(update)
    # craft a ClientLocalState with an exact, hand-set update
    state = local_train(base, client_id, data, round_id, local_cfg, StubRng(1))
    state.update = update.copy()
    return compress(state, CompressionConfig(), StubRng(1))


def test_aggregate_weighted_mean_matches_hand_computed():
    u1 = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0])
    u2 = np.array([-1.0, 2.0, 1.5, -3.0, 2.0, 1.0])
    compressed = [_compressed("client_a", u1), _compressed("client_b", u2)]
    weights = {"client_a": 1.0, "client_b": 3.0}
    cfg = AggregationConfig(rule="weighted_mean")
    state = aggregate(compressed, weights, cfg, StubRng(9))
    expected = 0.25 * u1 + 0.75 * u2
    np.testing.assert_allclose(state.aggregate, expected, rtol=0, atol=1e-15)
    assert state.weights == {"client_a": pytest.approx(0.25), "client_b": pytest.approx(0.75)}
    assert state.received_ids == ["client_a", "client_b"]
    assert state.accepted_ids == ["client_a", "client_b"]
    assert state.rejected_ids == []


def test_aggregate_uniform_mean():
    u1 = np.array([1.0, -2.0, 0.5, 3.0, 0.0, -1.0])
    u2 = np.array([-1.0, 2.0, 1.5, -3.0, 2.0, 1.0])
    compressed = [_compressed("client_a", u1), _compressed("client_b", u2)]
    cfg = AggregationConfig(rule="uniform_mean")
    state = aggregate(compressed, {"client_a": 99.0, "client_b": 1.0}, cfg, StubRng(9))
    np.testing.assert_allclose(state.aggregate, 0.5 * (u1 + u2), rtol=0, atol=1e-15)
    assert state.weights == {"client_a": pytest.approx(0.5), "client_b": pytest.approx(0.5)}


def test_aggregate_other_rules_not_implemented():
    compressed = [_compressed("client_a", np.zeros(6))]
    for rule in ("median", "trimmed_mean"):
        with pytest.raises(NotImplementedError):
            aggregate(compressed, {"client_a": 1.0}, AggregationConfig(rule=rule), StubRng(9))


def test_evaluate_on_separable_data():
    rng = np.random.default_rng(0)
    centers = np.array([[3.0] * 5, [-3.0] * 5])
    y = np.repeat([0, 1], 50)
    x = centers[y] + rng.normal(0, 0.5, size=(100, 5))
    params = np.zeros(2 * 6, dtype=np.float64)  # 2 classes, 5 features + bias
    params[:10] = np.concatenate([centers[0], centers[1]]) / 5.0  # decent W
    outcome = evaluate(params, ClientData(x=x.astype(np.float64), y=y))
    assert outcome.metrics["accuracy"] == pytest.approx(1.0)
    assert outcome.metrics["loss"] > 0.0
    assert set(outcome.per_class) == {"0", "1"}
    assert outcome.per_class["0"]["accuracy"] == pytest.approx(1.0)
    assert outcome.per_class["1"]["accuracy"] == pytest.approx(1.0)
    assert outcome.model_hash == hashlib.sha256(params.tobytes()).hexdigest()


# --- CONTRACTS section 3 stream-registry compliance (T2-F finding 5) ---


def test_select_clients_requests_only_client_selection_stream():
    pool = [f"client_{i}" for i in range(10)]
    spy = SpyRng(7, allowed={"client_selection"})
    select_clients(pool, 0, SelectionConfig(clients_per_round=4), spy)
    assert spy.requested == ["client_selection"]


def test_local_train_requests_only_client_dataloader_stream(client_data, params):
    spy = SpyRng(11, allowed={"client.client_0.dataloader"})
    cfg = LocalConfig(lr=0.5, local_steps=3, batch_size=8)
    local_train(params, "client_0", client_data, 0, cfg, spy)
    assert spy.requested == ["client.client_0.dataloader"]


def test_select_clients_unaffected_by_unrelated_streams():
    pool = [f"client_{i}" for i in range(10)]
    cfg = SelectionConfig(clients_per_round=4)
    fresh = select_clients(pool, 0, cfg, Rng(7))
    rng = Rng(7)
    rng.stream("global_init").normal(size=8)
    rng.stream("aggregation").integers(0, 100, size=3)
    rng.stream("client.client_0.dataloader").random(5)
    disturbed = select_clients(pool, 0, cfg, rng)
    assert fresh.selected_ids == disturbed.selected_ids


def test_select_clients_changes_when_client_selection_preconsumed():
    pool = [f"client_{i}" for i in range(10)]
    cfg = SelectionConfig(clients_per_round=4)
    fresh = select_clients(pool, 0, cfg, Rng(7))
    rng = Rng(7)
    rng.stream("client_selection").integers(0, 100)
    disturbed = select_clients(pool, 0, cfg, rng)
    assert fresh.selected_ids != disturbed.selected_ids


def test_local_train_unaffected_by_unrelated_streams(client_data, params):
    cfg = LocalConfig(lr=0.5, local_steps=4, batch_size=8)
    fresh = local_train(params, "client_0", client_data, 0, cfg, Rng(11))
    rng = Rng(11)
    rng.stream("global_init").normal(size=8)
    rng.stream("client.client_1.dataloader").integers(0, 10, size=4)
    disturbed = local_train(params, "client_0", client_data, 0, cfg, rng)
    np.testing.assert_array_equal(fresh.update, disturbed.update)


def test_local_train_changes_when_dataloader_stream_preconsumed(client_data, params):
    cfg = LocalConfig(lr=0.5, local_steps=4, batch_size=8)
    fresh = local_train(params, "client_0", client_data, 0, cfg, Rng(11))
    rng = Rng(11)
    rng.stream("client.client_0.dataloader").integers(0, 10)
    disturbed = local_train(params, "client_0", client_data, 0, cfg, rng)
    assert not np.array_equal(fresh.update, disturbed.update)


# --- rng_state snapshots at consuming stages (T2-F finding 3) ---


def test_select_clients_snapshots_consumed_stream_state():
    rng = Rng(7)
    pool = [f"client_{i}" for i in range(10)]
    state = select_clients(pool, 0, SelectionConfig(clients_per_round=4), rng)
    assert set(state.rng_state) == {"client_selection"}
    # Any-typed dict fields must stay JSON-native for the recorder.
    assert json.loads(json.dumps(state.rng_state)) == state.rng_state
    # Restoring the snapshot reproduces the stream's exact next draw.
    snapshot = copy.deepcopy(state.rng_state["client_selection"])
    expected = rng.stream("client_selection").integers(0, 2**32, size=8)
    restored = np.random.default_rng(0)
    restored.bit_generator.state = snapshot
    np.testing.assert_array_equal(restored.integers(0, 2**32, size=8), expected)


def test_local_train_snapshots_consumed_stream_state(client_data, params):
    rng = Rng(11)
    cfg = LocalConfig(lr=0.5, local_steps=3, batch_size=8)
    state = local_train(params, "client_0", client_data, 0, cfg, rng)
    assert set(state.rng_state) == {"client.client_0.dataloader"}
    assert json.loads(json.dumps(state.rng_state)) == state.rng_state
    snapshot = copy.deepcopy(state.rng_state["client.client_0.dataloader"])
    expected = rng.stream("client.client_0.dataloader").integers(0, 2**32, size=8)
    restored = np.random.default_rng(0)
    restored.bit_generator.state = snapshot
    np.testing.assert_array_equal(restored.integers(0, 2**32, size=8), expected)


# --- aggregate() boundary validation (T2-F finding 6) ---


def test_aggregate_rejects_zero_total_weight():
    compressed = [_compressed("client_a", np.ones(6)), _compressed("client_b", np.ones(6))]
    with pytest.raises(ValueError, match="positive"):
        aggregate(
            compressed,
            {"client_a": 0.0, "client_b": 0.0},
            AggregationConfig(rule="weighted_mean"),
            StubRng(9),
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_aggregate_rejects_non_finite_weight(bad):
    compressed = [_compressed("client_a", np.ones(6)), _compressed("client_b", np.ones(6))]
    with pytest.raises(ValueError, match="finite"):
        aggregate(
            compressed,
            {"client_a": bad, "client_b": 1.0},
            AggregationConfig(rule="weighted_mean"),
            StubRng(9),
        )


def test_aggregate_rejects_negative_weight():
    compressed = [_compressed("client_a", np.ones(6)), _compressed("client_b", np.ones(6))]
    with pytest.raises(ValueError, match="nonnegative"):
        aggregate(
            compressed,
            {"client_a": -1.0, "client_b": 2.0},
            AggregationConfig(rule="weighted_mean"),
            StubRng(9),
        )


def test_aggregate_rejects_missing_and_extra_weights():
    compressed = [_compressed("client_a", np.ones(6)), _compressed("client_b", np.ones(6))]
    cfg = AggregationConfig(rule="weighted_mean")
    with pytest.raises(ValueError, match="cover exactly"):
        aggregate(compressed, {"client_a": 1.0}, cfg, StubRng(9))
    with pytest.raises(ValueError, match="cover exactly"):
        aggregate(
            compressed,
            {"client_a": 1.0, "client_b": 1.0, "client_c": 1.0},
            cfg,
            StubRng(9),
        )


def test_aggregate_rejects_duplicate_client_ids():
    compressed = [_compressed("client_a", np.ones(6)), _compressed("client_a", np.zeros(6))]
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(
            compressed,
            {"client_a": 1.0},
            AggregationConfig(rule="weighted_mean"),
            StubRng(9),
        )


# --- evaluate() log-loss correctness and stability (T2-F finding 8) ---


def test_evaluate_hand_computed_mean_cross_entropy():
    # 3 classes, 1 feature; W = 0 and b = [ln2, 0, -ln2] gives probs [4/7, 2/7, 1/7].
    params = np.array([0.0, 0.0, 0.0, np.log(2.0), 0.0, -np.log(2.0)])
    x = np.array([[1.0], [-2.0]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)
    outcome = evaluate(params, ClientData(x=x, y=y))
    # mean (not sum) of -log(4/7) and -log(2/7)
    expected = 0.5 * (np.log(7.0 / 4.0) + np.log(7.0 / 2.0))
    assert outcome.metrics["loss"] == pytest.approx(expected, rel=1e-12)
    assert outcome.metrics["accuracy"] == pytest.approx(0.5)
    assert outcome.per_class["0"]["accuracy"] == pytest.approx(1.0)
    assert outcome.per_class["1"]["accuracy"] == pytest.approx(0.0)
    assert outcome.per_class["2"]["accuracy"] == 0.0  # absent class: zero convention


def test_evaluate_extreme_finite_logits_return_finite_loss():
    # 2 classes, 1 feature; logits reach +/-1e300 — max-subtracted log-softmax
    # must stay finite instead of overflowing through exp().
    params = np.array([1e300, -1e300, 0.0, 0.0])
    x = np.array([[1.0], [-1.0]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)
    outcome = evaluate(params, ClientData(x=x, y=y))
    assert np.isfinite(outcome.metrics["loss"])
    assert outcome.metrics["loss"] >= 0.0
    assert outcome.metrics["accuracy"] == pytest.approx(1.0)
