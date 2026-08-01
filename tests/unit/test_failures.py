"""Unit tests for the four failure injectors (Task T4, docs/tasks/T4).

Covers, per injector: identity outside the active window (object-equal, new
objects, inputs unmutated), active behavior vs. parameters, determinism for
the same spec+seed, and the CONTRACTS §3 rule that the ONLY rng stream an
injector may consume is ``failure.<stage>``.
"""
import hashlib

import numpy as np
import pytest

from falcon.failures import FailureInjector, build_injector
from falcon.failures.aggregation import WrongSampleWeightsInjector
from falcon.failures.compression import AggressiveTopKInjector
from falcon.failures.local import LrMisconfigInjector
from falcon.failures.selection import MinorityExclusionInjector
from falcon.pipeline.stages import aggregate, compress
from falcon.pipeline.synthetic_data import ClientData
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    ClientLocalState,
    CompressionConfig,
    FailureSpecification,
    LocalConfig,
)


class SpyRng:
    """Wraps the production Rng, recording every requested stream name.

    If ``allowed`` is given, requesting a stream outside that set fails
    immediately — injectors must touch only their ``failure.<stage>`` stream.
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


def _spec(stage, type_, active=(0, 0), **parameters) -> FailureSpecification:
    return FailureSpecification(
        stage=stage, type=type_, active_rounds=active, parameters=parameters
    )


@pytest.fixture
def partition() -> dict[str, ClientData]:
    """10-sample clients; only client_0 is class-1-heavy (share 0.4 vs global 1/6)."""
    return {
        "client_0": ClientData(
            x=np.zeros((10, 3)), y=np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        ),
        "client_1": ClientData(
            x=np.zeros((10, 3)), y=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ),
        "client_2": ClientData(x=np.zeros((10, 3)), y=np.zeros(10, dtype=np.int64)),
    }


def _local_state(update: np.ndarray, client_id: str = "client_0") -> ClientLocalState:
    return ClientLocalState(
        round_id=0,
        client_id=client_id,
        base_model_hash="0" * 64,
        update=update.astype(np.float64),
        num_examples=10,
        num_steps=1,
        loss_history=[],
    )


# --- base class / dispatcher ------------------------------------------------


def test_active_window_is_inclusive(partition):
    injector = FailureInjector(_spec("local", "lr_misconfig", active=(2, 4)), partition, Rng(0))
    assert [injector.active(r) for r in range(7)] == [
        False, False, True, True, True, False, False,
    ]


def test_base_transforms_are_identity_even_when_active(partition):
    injector = FailureInjector(_spec("local", "lr_misconfig", active=(0, 99)), partition, Rng(0))
    pool = ["client_0", "client_1"]
    weights = {"client_0": 10.0, "client_1": 20.0}
    local = LocalConfig(lr=0.1, local_steps=5, batch_size=32)
    comp = CompressionConfig(kind="identity")
    assert injector.candidate_pool(pool, 3) == pool
    assert injector.weights(weights, 3) == weights
    assert injector.local_cfg("client_0", local, 3) == local
    assert injector.compression_cfg("client_0", comp, 3) == comp


def test_build_injector_dispatches(partition):
    cases = [
        (_spec("selection", "minority_exclusion", target_class=1, exclusion_probability=0.5), MinorityExclusionInjector),
        (_spec("local", "lr_misconfig", affected_clients=["client_0"], lr_multiplier=10.0), LrMisconfigInjector),
        (_spec("compression", "aggressive_topk", k_ratio=0.1), AggressiveTopKInjector),
        (_spec("aggregation", "wrong_sample_weights", mode="uniform"), WrongSampleWeightsInjector),
    ]
    for spec, cls in cases:
        assert isinstance(build_injector(spec, partition, Rng(0)), cls)


def test_build_injector_rejects_unknown(partition):
    with pytest.raises(ValueError, match="unknown failure injector"):
        build_injector(_spec("evaluation", "nope"), partition, Rng(0))
    with pytest.raises(ValueError, match="unknown failure injector"):
        build_injector(_spec("selection", "lr_misconfig"), partition, Rng(0))


def test_stage_guard_rejects_mismatched_spec(partition):
    with pytest.raises(ValueError, match="stage"):
        MinorityExclusionInjector(
            _spec("local", "minority_exclusion", target_class=1, exclusion_probability=1.0),
            partition,
            Rng(0),
        )


# --- S1: selection / minority_exclusion --------------------------------------


def _selection_injector(partition, seed=7, active=(1, 3), p=1.0):
    return MinorityExclusionInjector(
        _spec("selection", "minority_exclusion", active=active,
              target_class=1, exclusion_probability=p),
        partition,
        Rng(seed),
    )


def test_minority_heavy_detection(partition):
    injector = _selection_injector(partition)
    assert injector.minority_heavy_clients == frozenset({"client_0"})


def test_exclusion_inactive_rounds_are_identity(partition):
    injector = _selection_injector(partition, active=(2, 3))
    pool = ["client_0", "client_1", "client_2"]
    for round_id in (0, 1, 4, 10):
        out = injector.candidate_pool(pool, round_id)
        assert out == pool
        assert out is not pool
    assert pool == ["client_0", "client_1", "client_2"]  # input unmutated


def test_exclusion_p1_removes_all_minority_heavy(partition):
    injector = _selection_injector(partition, p=1.0)
    pool = ["client_0", "client_1", "client_2"]
    for round_id in (1, 2, 3):
        assert injector.candidate_pool(pool, round_id) == ["client_1", "client_2"]


def test_exclusion_p0_removes_nothing(partition):
    injector = _selection_injector(partition, p=0.0)
    pool = ["client_0", "client_1", "client_2"]
    assert injector.candidate_pool(pool, 2) == pool


def test_exclusion_deterministic_same_seed(partition):
    a = _selection_injector(partition, seed=11, p=0.5)
    b = _selection_injector(partition, seed=11, p=0.5)
    pool = ["client_0", "client_1", "client_2"]
    for round_id in (1, 2, 3):
        assert a.candidate_pool(pool, round_id) == b.candidate_pool(pool, round_id)


def test_exclusion_uses_only_failure_selection_stream(partition):
    spy = SpyRng(7, allowed={"failure.selection"})
    injector = MinorityExclusionInjector(
        _spec("selection", "minority_exclusion", active=(0, 99),
              target_class=1, exclusion_probability=0.5),
        partition,
        spy,
    )
    injector.candidate_pool(["client_0", "client_1"], 0)
    injector.candidate_pool(["client_0", "client_1"], 1)
    assert spy.requested  # active rounds did draw
    assert set(spy.requested) == {"failure.selection"}


def test_exclusion_inactive_rounds_draw_nothing(partition):
    spy = SpyRng(7, allowed={"failure.selection"})
    injector = MinorityExclusionInjector(
        _spec("selection", "minority_exclusion", active=(5, 6),
              target_class=1, exclusion_probability=0.5),
        partition,
        spy,
    )
    injector.candidate_pool(["client_0", "client_1"], 0)
    assert spy.requested == []


def test_exclusion_rejects_bad_probability(partition):
    with pytest.raises(ValueError, match="exclusion_probability"):
        _selection_injector(partition, p=1.5)


# --- L1: local / lr_misconfig ------------------------------------------------


def _lr_injector(partition, seed=3, active=(1, 2), **params):
    defaults = {"affected_clients": ["client_0", "client_2"], "lr_multiplier": 10.0}
    defaults.update(params)
    return LrMisconfigInjector(
        _spec("local", "lr_misconfig", active=active, **defaults), partition, Rng(seed)
    )


def test_lr_inactive_rounds_are_identity(partition):
    injector = _lr_injector(partition)
    cfg = LocalConfig(lr=0.1, local_steps=5, batch_size=32)
    for round_id in (0, 3, 9):
        out = injector.local_cfg("client_0", cfg, round_id)
        assert out == cfg
        assert out is not cfg
    assert cfg.lr == 0.1  # input unmutated


def test_lr_scaled_for_affected_only(partition):
    injector = _lr_injector(partition, lr_multiplier=10.0)
    cfg = LocalConfig(lr=0.1, local_steps=5, batch_size=32)
    for round_id in (1, 2):
        hit = injector.local_cfg("client_0", cfg, round_id)
        assert hit.lr == pytest.approx(1.0)
        assert hit.local_steps == cfg.local_steps and hit.batch_size == cfg.batch_size
        miss = injector.local_cfg("client_1", cfg, round_id)
        assert miss == cfg


def test_lr_fraction_selects_once_deterministically(partition):
    a = _lr_injector(partition, seed=5, affected_clients=None, fraction=0.5)
    b = _lr_injector(partition, seed=5, affected_clients=None, fraction=0.5)
    assert a.affected_clients == b.affected_clients
    assert len(a.affected_clients) == 2  # round(0.5 * 3)
    assert a.affected_clients <= frozenset(partition)


def test_lr_fraction_uses_only_failure_local_stream(partition):
    spy = SpyRng(5, allowed={"failure.local"})
    injector = LrMisconfigInjector(
        _spec("local", "lr_misconfig", active=(0, 99), fraction=0.5, lr_multiplier=2.0),
        partition,
        spy,
    )
    injector.local_cfg("client_0", LocalConfig(lr=0.1, local_steps=1, batch_size=1), 0)
    assert set(spy.requested) == {"failure.local"}


def test_lr_explicit_clients_draw_nothing(partition):
    spy = SpyRng(5, allowed={"failure.local"})
    injector = LrMisconfigInjector(
        _spec("local", "lr_misconfig", active=(0, 99),
              affected_clients=["client_0"], lr_multiplier=2.0),
        partition,
        spy,
    )
    injector.local_cfg("client_0", LocalConfig(lr=0.1, local_steps=1, batch_size=1), 0)
    assert spy.requested == []


def test_lr_requires_exactly_one_selector(partition):
    base = {"lr_multiplier": 2.0}
    with pytest.raises(ValueError, match="exactly one"):
        LrMisconfigInjector(_spec("local", "lr_misconfig", **base), partition, Rng(0))
    with pytest.raises(ValueError, match="exactly one"):
        LrMisconfigInjector(
            _spec("local", "lr_misconfig", affected_clients=["client_0"], fraction=0.5, **base),
            partition,
            Rng(0),
        )
    with pytest.raises(ValueError, match="not in partition"):
        _lr_injector(partition, affected_clients=["ghost"])


# --- C1: compression / aggressive_topk (+ stages.compress topk) --------------


def _topk_injector(partition, active=(1, 2), **params):
    defaults = {"k_ratio": 0.1}
    defaults.update(params)
    return AggressiveTopKInjector(
        _spec("compression", "aggressive_topk", active=active, **defaults),
        partition,
        Rng(1),
    )


def test_topk_injector_inactive_rounds_are_identity(partition):
    injector = _topk_injector(partition)
    cfg = CompressionConfig(kind="identity")
    for round_id in (0, 3):
        out = injector.compression_cfg("client_0", cfg, round_id)
        assert out == cfg
        assert out is not cfg
    assert cfg.kind == "identity"  # input unmutated


def test_topk_injector_swaps_cfg_when_active(partition):
    injector = _topk_injector(partition, k_ratio=0.05)
    cfg = CompressionConfig(kind="identity", parameters={"keep": "me-not"})
    for round_id in (1, 2):
        out = injector.compression_cfg("client_0", cfg, round_id)
        assert out.kind == "topk"
        assert out.parameters == {"k_ratio": 0.05}
        assert out is not cfg


def test_topk_injector_respects_affected_clients(partition):
    injector = _topk_injector(partition, affected_clients=["client_1"])
    cfg = CompressionConfig(kind="identity")
    assert injector.compression_cfg("client_0", cfg, 1).kind == "identity"
    assert injector.compression_cfg("client_1", cfg, 1).kind == "topk"


def test_topk_injector_defaults_to_all_clients(partition):
    injector = _topk_injector(partition)
    assert injector.affected_clients == frozenset(partition)


def test_topk_injector_draws_no_randomness(partition):
    spy = SpyRng(1, allowed={"failure.compression"})
    injector = AggressiveTopKInjector(
        _spec("compression", "aggressive_topk", active=(0, 99), k_ratio=0.1),
        partition,
        spy,
    )
    injector.compression_cfg("client_0", CompressionConfig(), 0)
    assert spy.requested == []  # top-k is deterministic: no stream consumed


def test_topk_injector_rejects_bad_ratio(partition):
    for bad in (0.0, -0.5, 1.5, float("nan")):
        with pytest.raises(ValueError, match="k_ratio"):
            _topk_injector(partition, k_ratio=bad)


def test_compress_topk_keeps_exactly_ceil_ratio_times_n():
    update = np.arange(1, 11, dtype=np.float64)  # distinct magnitudes, n = 10
    state = _local_state(update)
    comp = compress(state, CompressionConfig(kind="topk", parameters={"k_ratio": 0.26}), Rng(0))
    assert int(np.count_nonzero(comp.update)) == 3  # ceil(0.26 * 10)
    np.testing.assert_array_equal(comp.update, [0] * 7 + [8.0, 9.0, 10.0])
    assert comp.bytes_transmitted == 3 * 8 + 3 * 4
    assert comp.compression_params == {"k_ratio": 0.26}
    assert comp.uncompressed_hash == hashlib.sha256(update.tobytes()).hexdigest()
    # input state not mutated, output not aliased
    np.testing.assert_array_equal(state.update, update)
    assert comp.update is not state.update


def test_compress_topk_tie_break_keeps_larger_index():
    # three coordinates tie at |1.0| (indices 0, 1, 2); k = 2 keeps indices 1, 2.
    update = np.array([1.0, -1.0, 1.0, 0.5])
    comp = compress(
        _local_state(update),
        CompressionConfig(kind="topk", parameters={"k_ratio": 0.5}),
        Rng(0),
    )
    np.testing.assert_array_equal(comp.update, [0.0, -1.0, 1.0, 0.0])


def test_compress_topk_is_deterministic():
    update = np.array([3.0, -1.0, 2.0, -4.0, 0.0, 1.5])
    cfg = CompressionConfig(kind="topk", parameters={"k_ratio": 0.34})
    a = compress(_local_state(update), cfg, Rng(9))
    b = compress(_local_state(update), cfg, Rng(9))
    np.testing.assert_array_equal(a.update, b.update)
    assert a.bytes_transmitted == b.bytes_transmitted


def test_compress_topk_ratio_one_round_trips():
    update = np.array([1.0, -2.0, 3.0])
    comp = compress(
        _local_state(update), CompressionConfig(kind="topk", parameters={"k_ratio": 1.0}), Rng(0)
    )
    np.testing.assert_array_equal(comp.update, update)


def test_compress_topk_rejects_bad_ratio():
    for bad in (0.0, 1.5, float("inf")):
        with pytest.raises(ValueError, match="k_ratio"):
            compress(
                _local_state(np.ones(4)),
                CompressionConfig(kind="topk", parameters={"k_ratio": bad}),
                Rng(0),
            )


# --- A1: aggregation / wrong_sample_weights ----------------------------------


def _weights_injector(partition, mode, active=(1, 2), seed=13):
    return WrongSampleWeightsInjector(
        _spec("aggregation", "wrong_sample_weights", active=active, mode=mode),
        partition,
        Rng(seed),
    )


def test_weights_inactive_rounds_are_identity(partition):
    injector = _weights_injector(partition, "corrupted")
    weights = {"client_0": 30.0, "client_1": 10.0}
    for round_id in (0, 3, 8):
        out = injector.weights(weights, round_id)
        assert out == weights
        assert out is not weights
    assert weights == {"client_0": 30.0, "client_1": 10.0}  # input unmutated


def test_weights_uniform_mode(partition):
    injector = _weights_injector(partition, "uniform")
    out = injector.weights({"client_1": 10.0, "client_0": 30.0, "client_2": 5.0}, 1)
    assert out == {"client_0": 1.0, "client_1": 1.0, "client_2": 1.0}


def test_weights_swapped_mode_reverses_across_sorted_ids(partition):
    injector = _weights_injector(partition, "swapped")
    out = injector.weights({"client_2": 5.0, "client_0": 30.0, "client_1": 10.0}, 1)
    # sorted ids [client_0, client_1, client_2] values [30, 10, 5] reversed
    assert out == {"client_0": 5.0, "client_1": 10.0, "client_2": 30.0}


def test_weights_corrupted_mode_log_uniform_factors(partition):
    weights = {"client_0": 100.0, "client_1": 100.0, "client_2": 100.0}
    out = _weights_injector(partition, "corrupted").weights(weights, 1)
    assert set(out) == set(weights)
    for cid, value in out.items():
        factor = value / weights[cid]
        assert 0.1 <= factor <= 10.0
        assert factor != 1.0
    # deterministic for the same spec+seed
    again = _weights_injector(partition, "corrupted").weights(weights, 1)
    assert out == again


def test_weights_corrupted_uses_only_failure_aggregation_stream(partition):
    spy = SpyRng(13, allowed={"failure.aggregation"})
    injector = WrongSampleWeightsInjector(
        _spec("aggregation", "wrong_sample_weights", active=(0, 99), mode="corrupted"),
        partition,
        spy,
    )
    injector.weights({"client_0": 1.0, "client_1": 2.0}, 0)
    assert set(spy.requested) == {"failure.aggregation"}
    assert len(spy.requested) == 1  # one stream, one draw per client


def test_weights_uniform_and_swapped_draw_nothing(partition):
    for mode in ("uniform", "swapped"):
        spy = SpyRng(13, allowed={"failure.aggregation"})
        injector = WrongSampleWeightsInjector(
            _spec("aggregation", "wrong_sample_weights", active=(0, 99), mode=mode),
            partition,
            spy,
        )
        injector.weights({"client_0": 1.0}, 0)
        assert spy.requested == []


def test_weights_corrupted_feeds_aggregate_renormalization(partition):
    """Corrupted weights still flow through aggregate()'s usual renormalization."""
    weights = {"client_0": 100.0, "client_1": 20.0}
    corrupted = _weights_injector(partition, "corrupted").weights(weights, 1)
    compressed = [
        compress(_local_state(np.ones(4), cid), CompressionConfig(), Rng(0))
        for cid in ("client_0", "client_1")
    ]
    state = aggregate(compressed, corrupted, AggregationConfig(rule="weighted_mean"), Rng(0))
    assert sum(state.weights.values()) == pytest.approx(1.0)


def test_weights_rejects_unknown_mode(partition):
    with pytest.raises(ValueError, match="mode"):
        _weights_injector(partition, "sideways")


# --- cross-cutting determinism (same spec + seed -> identical transforms) ----


def test_same_spec_and_seed_give_identical_transform_sequences(partition):
    def sequence(seed):
        injector = build_injector(
            _spec("aggregation", "wrong_sample_weights", active=(0, 3), mode="corrupted"),
            partition,
            Rng(seed),
        )
        return [injector.weights({"client_0": 3.0, "client_1": 7.0}, r) for r in range(4)]

    assert sequence(21) == sequence(21)
    assert sequence(21) != sequence(22)
