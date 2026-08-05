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
from falcon.failures.aggregation import AggressiveClippingInjector, WrongSampleWeightsInjector
from falcon.failures.compression import AggressiveQuantizationInjector, AggressiveTopKInjector
from falcon.failures.local import (
    LabelCorruptionInjector,
    LrMisconfigInjector,
    ModelPoisoningInjector,
)
from falcon.failures.selection import AvailabilityBiasInjector, MinorityExclusionInjector
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
    assert injector.local_data("client_0", partition["client_0"], 3) is partition["client_0"]
    state = _local_state(np.ones(4))
    assert injector.local_state("client_0", state, 3) is state
    assert injector.compression_cfg("client_0", comp, 3) == comp
    assert injector.aggregation_cfg(AggregationConfig(), 3) == AggregationConfig()


def test_build_injector_dispatches(partition):
    cases = [
        (_spec("selection", "minority_exclusion", target_class=1, exclusion_probability=0.5), MinorityExclusionInjector),
        (_spec("selection", "availability_bias", biased_fraction=0.5), AvailabilityBiasInjector),
        (_spec("local", "lr_misconfig", affected_clients=["client_0"], lr_multiplier=10.0), LrMisconfigInjector),
        (_spec("local", "label_corruption", fraction_clients=0.5, flip_probability=0.5), LabelCorruptionInjector),
        (_spec("local", "model_poisoning", fraction_clients=0.5), ModelPoisoningInjector),
        (_spec("compression", "aggressive_topk", k_ratio=0.1), AggressiveTopKInjector),
        (_spec("compression", "aggressive_quantization"), AggressiveQuantizationInjector),
        (_spec("aggregation", "wrong_sample_weights", mode="uniform"), WrongSampleWeightsInjector),
        (_spec("aggregation", "aggressive_clipping"), AggressiveClippingInjector),
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


def test_exclusion_rejects_non_finite_probability(partition):
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="exclusion_probability"):
            _selection_injector(partition, p=bad)


# --- S2: selection / availability_bias -------------------------------------


def _availability_injector(partition, seed=7, active=(1, 20), **params):
    defaults = {"biased_fraction": 2 / 3, "availability": 0.5}
    defaults.update(params)
    return AvailabilityBiasInjector(
        _spec("selection", "availability_bias", active=active, **defaults),
        partition,
        Rng(seed),
    )


def test_availability_bias_deterministic_and_client_order_independent(partition):
    forward = _availability_injector(partition, seed=11)
    reverse = _availability_injector(partition, seed=11)
    pool = sorted(partition)
    for round_id in range(1, 21):
        assert forward.candidate_pool(pool, round_id) == reverse.candidate_pool(
            list(reversed(pool)), round_id
        )


def test_availability_bias_reference_is_untouched(partition):
    injector = _availability_injector(partition, active=(2, 3))
    pool = ["client_2", "client_0", "client_1"]
    out = injector.candidate_pool(pool, 1)
    assert out == pool
    assert out is not pool
    assert pool == ["client_2", "client_0", "client_1"]


def test_availability_bias_exclusion_is_stochastic(partition):
    injector = _availability_injector(partition, active=(0, 99))
    presence = [
        "client_0" in injector.candidate_pool(sorted(partition), round_id)
        for round_id in range(100)
    ]
    assert any(presence)
    assert not all(presence)
    assert injector.biased_clients == frozenset({"client_0", "client_1"})


def test_availability_bias_uses_per_client_round_streams(partition):
    allowed = {
        "failure.selection.client_0.round.1",
        "failure.selection.client_1.round.1",
    }
    spy = SpyRng(7, allowed=allowed)
    injector = AvailabilityBiasInjector(
        _spec(
            "selection",
            "availability_bias",
            active=(1, 1),
            biased_fraction=2 / 3,
            availability=0.5,
        ),
        partition,
        spy,
    )
    injector.candidate_pool(sorted(partition), 1)
    assert set(spy.requested) == allowed


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


def test_lr_rejects_non_finite_multiplier(partition):
    """T8-F finding 6: a NaN/inf multiplier would poison every stage state."""
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="lr_multiplier"):
            _lr_injector(partition, lr_multiplier=bad)


def test_lr_rejects_non_finite_fraction(partition):
    with pytest.raises(ValueError, match="fraction"):
        _lr_injector(partition, affected_clients=None, fraction=float("nan"))
    with pytest.raises(ValueError, match="fraction"):
        _lr_injector(partition, affected_clients=None, fraction=float("inf"))


# --- L4: local / label_corruption ------------------------------------------


def _label_injector(partition, seed=17, active=(1, 2), **params):
    defaults = {"fraction_clients": 2 / 3, "flip_probability": 0.5}
    defaults.update(params)
    return LabelCorruptionInjector(
        _spec("local", "label_corruption", active=active, **defaults),
        partition,
        Rng(seed),
    )


def test_label_corruption_is_deterministic_and_targets_sorted_prefix(partition):
    first = _label_injector(partition, seed=19)
    second = _label_injector(partition, seed=19)
    assert first.affected_clients == frozenset({"client_0", "client_1"})
    for cid in sorted(partition):
        a = first.local_data(cid, partition[cid], 1)
        b = second.local_data(cid, partition[cid], 1)
        np.testing.assert_array_equal(a.y, b.y)
        assert a.x is partition[cid].x
    assert first.local_data("client_2", partition["client_2"], 1) is partition["client_2"]


def test_label_corruption_client_order_independent(partition):
    forward = _label_injector(partition, seed=23)
    reverse = _label_injector(partition, seed=23)
    forward_labels = {
        cid: forward.local_data(cid, partition[cid], 1).y
        for cid in sorted(partition)
    }
    reverse_labels = {
        cid: reverse.local_data(cid, partition[cid], 1).y
        for cid in reversed(sorted(partition))
    }
    for cid in partition:
        np.testing.assert_array_equal(forward_labels[cid], reverse_labels[cid])


def test_label_corruption_reference_and_inactive_data_are_untouched(partition):
    injector = _label_injector(partition, active=(2, 3), flip_probability=1.0)
    data = partition["client_0"]
    assert injector.local_data("client_0", data, 1) is data
    np.testing.assert_array_equal(data.y, partition["client_0"].y)


def test_label_corruption_uses_per_client_round_streams(partition):
    allowed = {
        "failure.local.client_0.round.1",
        "failure.local.client_1.round.1",
    }
    spy = SpyRng(17, allowed=allowed)
    injector = LabelCorruptionInjector(
        _spec(
            "local",
            "label_corruption",
            active=(1, 1),
            fraction_clients=2 / 3,
            flip_probability=0.5,
        ),
        partition,
        spy,
    )
    for cid in sorted(partition):
        injector.local_data(cid, partition[cid], 1)
    assert set(spy.requested) == allowed


# --- L5: local / model_poisoning -------------------------------------------


def _poisoning_injector(partition, seed=17, active=(1, 2), **params):
    defaults = {"fraction_clients": 2 / 3, "scale": 5.0}
    defaults.update(params)
    return ModelPoisoningInjector(
        _spec("local", "model_poisoning", active=active, **defaults),
        partition,
        Rng(seed),
    )


def test_model_poisoning_replaces_update_only(partition):
    injector = _poisoning_injector(partition, scale=3.0)
    state = _local_state(np.array([1.0, -2.0, 0.5]))
    poisoned = injector.local_state("client_0", state, 1)
    np.testing.assert_array_equal(poisoned.update, -3.0 * state.update)
    assert poisoned.model_copy(update={"update": state.update}) == state
    np.testing.assert_array_equal(state.update, [1.0, -2.0, 0.5])


def test_model_poisoning_reference_and_unaffected_states_are_untouched(partition):
    injector = _poisoning_injector(partition, active=(2, 3))
    affected = _local_state(np.ones(3), "client_0")
    unaffected = _local_state(np.ones(3), "client_2")
    assert injector.local_state("client_0", affected, 1) is affected
    assert injector.local_state("client_2", unaffected, 2) is unaffected


def test_model_poisoning_affected_set_is_deterministic(partition):
    first = _poisoning_injector(partition, seed=1)
    second = _poisoning_injector(partition, seed=99)
    assert first.affected_clients == second.affected_clients == frozenset(
        {"client_0", "client_1"}
    )


def test_aggressive_quantization_uses_severity_bits(partition):
    cfg = CompressionConfig(kind="identity")
    for severity, bits in ((1, 8), (2, 4), (3, 2)):
        spec = _spec("compression", "aggressive_quantization").model_copy(
            update={"severity": severity}
        )
        injector = AggressiveQuantizationInjector(spec, partition, Rng(1))
        out = injector.compression_cfg("client_0", cfg, 0)
        assert out.kind == "quantization"
        assert out.parameters == {"bits": bits}


def test_aggressive_clipping_uses_severity_norm(partition):
    cfg = AggregationConfig(rule="trimmed_mean", parameters={"beta": 0.2})
    for severity, clip_norm in ((1, 1.0), (2, 0.1), (3, 0.01)):
        spec = _spec("aggregation", "aggressive_clipping").model_copy(
            update={"severity": severity}
        )
        injector = AggressiveClippingInjector(spec, partition, Rng(1))
        out = injector.aggregation_cfg(cfg, 0)
        assert out.rule == "trimmed_mean"
        assert out.parameters == {"beta": 0.2, "clip_norm": clip_norm}


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


#: exact ``100 * 10 ** uniform(-1, 1)`` weights drawn from stream
#: ``failure.aggregation`` of ``Rng(13)`` in sorted-id order (finding 8: pin
#: the exact draw, not only range + repeatability)
_PINNED_CORRUPTED_WEIGHTS = {
    "client_0": 32.78828069651116,
    "client_1": 47.89096442259545,
    "client_2": 21.764116313365502,
}


def test_weights_corrupted_mode_pins_exact_rng_draws(partition):
    weights = {"client_0": 100.0, "client_1": 100.0, "client_2": 100.0}
    out = _weights_injector(partition, "corrupted").weights(weights, 1)
    assert out == _PINNED_CORRUPTED_WEIGHTS


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


# --- A1/T15: corrupted-mode intensity knob -----------------------------------


def _intensity_injector(partition, intensity, seed=13, active=(1, 2)):
    return WrongSampleWeightsInjector(
        _spec("aggregation", "wrong_sample_weights", active=active,
              mode="corrupted", intensity=intensity),
        partition,
        Rng(seed),
    )


def test_weights_corrupted_intensity_1_bit_identical(partition):
    """T15: explicit intensity=1.0 is the same stream, same draws as today."""
    weights = {"client_0": 100.0, "client_1": 100.0, "client_2": 100.0}
    explicit = _intensity_injector(partition, 1.0).weights(weights, 1)
    default = _weights_injector(partition, "corrupted").weights(weights, 1)
    assert explicit == default == _PINNED_CORRUPTED_WEIGHTS
    # and across rounds, not just the first draw
    for round_id in (1, 2):
        assert (
            _intensity_injector(partition, 1.0).weights(weights, round_id)
            == _weights_injector(partition, "corrupted").weights(weights, round_id)
        )


def test_weights_corrupted_intensity_spread_monotone(partition):
    """Higher intensity -> larger weight spread (same seed, same stream)."""
    weights = {f"client_{i}": 100.0 for i in range(64)}
    spreads = []
    for intensity in (0.25, 1.0, 2.0, 4.0):
        out = _intensity_injector(partition, intensity, active=(0, 0)).weights(weights, 0)
        factors = [out[cid] / weights[cid] for cid in weights]
        assert all(10.0 ** -intensity <= f <= 10.0 ** intensity for f in factors)
        spreads.append(max(np.log10(factors)) - min(np.log10(factors)))
    assert all(b > a for a, b in zip(spreads, spreads[1:])), spreads


def test_weights_corrupted_rejects_bad_intensity(partition):
    for bad in (0.0, -0.5, 4.5, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="intensity"):
            _intensity_injector(partition, bad)


def test_weights_uniform_and_swapped_reject_intensity(partition):
    """T15: the knob is corrupted-only; other modes fail loud, even at 1.0."""
    for mode in ("uniform", "swapped"):
        for intensity in (1.0, 2.0):
            with pytest.raises(ValueError, match="intensity"):
                WrongSampleWeightsInjector(
                    _spec("aggregation", "wrong_sample_weights",
                          mode=mode, intensity=intensity),
                    partition,
                    Rng(13),
                )


# --- A1/T17: biased-mode deterministic minority-heavy down-weighting ---------


def _biased_injector(partition, weight_multiplier=0.5, active=(1, 2), seed=13, rng=None):
    return WrongSampleWeightsInjector(
        _spec("aggregation", "wrong_sample_weights", active=active,
              mode="biased", weight_multiplier=weight_multiplier, target_class=1),
        partition,
        rng if rng is not None else Rng(seed),
    )


def test_weights_biased_targets_minority_heavy_only(partition):
    """Same targeting rule as minority_exclusion: only client_0 is class-1-heavy."""
    injector = _biased_injector(partition, weight_multiplier=0.5)
    assert injector.minority_heavy_clients == frozenset({"client_0"})
    weights = {"client_0": 30.0, "client_1": 10.0, "client_2": 5.0}
    for round_id in (1, 2):
        out = injector.weights(weights, round_id)
        assert out == {"client_0": 15.0, "client_1": 10.0, "client_2": 5.0}
    assert weights == {"client_0": 30.0, "client_1": 10.0, "client_2": 5.0}  # unmutated


def test_weights_biased_inactive_rounds_are_identity(partition):
    injector = _biased_injector(partition, weight_multiplier=0.1)
    weights = {"client_0": 30.0, "client_1": 10.0}
    for round_id in (0, 3, 8):
        out = injector.weights(weights, round_id)
        assert out == weights
        assert out is not weights


def test_weights_biased_multiplier_1_is_exact_noop(partition):
    """T17: multiplier 1.0 leaves every weight untouched, exactly."""
    injector = _biased_injector(partition, weight_multiplier=1.0)
    weights = {"client_0": 30.0, "client_1": 10.0, "client_2": 5.0}
    out = injector.weights(weights, 1)
    assert out == weights
    assert out is not weights
    for cid in weights:
        assert out[cid] == weights[cid]  # no float drift, not even 1 ulp


def test_weights_biased_deterministic_and_seed_independent(partition):
    """No RNG at all: the output cannot depend on the run seed."""
    weights = {"client_0": 30.0, "client_1": 10.0, "client_2": 5.0}
    a = _biased_injector(partition, weight_multiplier=0.1, seed=13)
    b = _biased_injector(partition, weight_multiplier=0.1, seed=99)
    for round_id in (1, 2):
        assert a.weights(weights, round_id) == b.weights(weights, round_id)


def test_weights_biased_draws_no_randomness(partition):
    spy = SpyRng(13, allowed={"failure.aggregation"})
    injector = _biased_injector(partition, weight_multiplier=0.1, active=(0, 99), rng=spy)
    injector.weights({"client_0": 1.0, "client_1": 2.0}, 0)
    assert spy.requested == []  # biased mode is deterministic: no stream consumed


def test_weights_biased_normalized_share_monotone_in_multiplier(partition):
    """Lower multiplier -> strictly lower normalized share for targeted clients."""
    weights = {"client_0": 30.0, "client_1": 10.0, "client_2": 5.0}
    shares = []
    for multiplier in (1.0, 0.5, 0.1):
        out = _biased_injector(partition, multiplier).weights(weights, 1)
        total = sum(out.values())
        shares.append(out["client_0"] / total)
    assert shares[0] > shares[1] > shares[2]


def test_weights_biased_rejects_bad_multiplier(partition):
    for bad in (0.0, -0.5, 1.5, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="weight_multiplier"):
            _biased_injector(partition, weight_multiplier=bad)


def test_weights_biased_requires_weight_multiplier(partition):
    with pytest.raises(ValueError, match="weight_multiplier"):
        WrongSampleWeightsInjector(
            _spec("aggregation", "wrong_sample_weights", mode="biased", target_class=1),
            partition,
            Rng(13),
        )


def test_weights_biased_requires_target_class(partition):
    with pytest.raises(ValueError, match="target_class"):
        WrongSampleWeightsInjector(
            _spec("aggregation", "wrong_sample_weights",
                  mode="biased", weight_multiplier=0.5),
            partition,
            Rng(13),
        )


def test_weights_biased_rejects_intensity(partition):
    with pytest.raises(ValueError, match="intensity"):
        WrongSampleWeightsInjector(
            _spec("aggregation", "wrong_sample_weights", mode="biased",
                  weight_multiplier=0.5, target_class=1, intensity=1.0),
            partition,
            Rng(13),
        )


def test_weights_other_modes_reject_biased_knobs(partition):
    """T17: weight_multiplier/target_class are biased-only; other modes fail loud."""
    for mode in ("uniform", "swapped", "corrupted"):
        for knob, value in (("weight_multiplier", 0.5), ("target_class", 1)):
            with pytest.raises(ValueError, match=knob):
                WrongSampleWeightsInjector(
                    _spec("aggregation", "wrong_sample_weights",
                          mode=mode, **{knob: value}),
                    partition,
                    Rng(13),
                )


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
