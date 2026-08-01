"""Unit tests for the passive/terminal baselines (Task T8, Plan §19.1-19.2)."""
import numpy as np
import pytest

from falcon.baselines import (
    INTERVENABLE_STAGES,
    NearestCentroidStageClassifier,
    passive_localize,
    passive_stage_scores,
    terminal_features,
)
from falcon.pipeline.runner import run
from falcon.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    AggregationState,
    ClientLocalState,
    CompressionConfig,
    CompressionState,
    DatasetConfig,
    FailureSpecification,
    LocalConfig,
    OutcomeState,
    RunConfig,
    SelectionConfig,
    SelectionState,
)

CLIENTS = ["c0", "c1", "c2"]
HASH = "0" * 64


# --------------------------------------------------------------------------
# helpers: hand-built recorded runs (real Recorder, tiny synthetic states)
# --------------------------------------------------------------------------

def _selection(round_id, selected=CLIENTS):
    return SelectionState(
        round_id=round_id,
        candidate_ids=CLIENTS,
        selected_ids=list(selected),
        sampling_probs={c: 1.0 / len(CLIENTS) for c in CLIENTS},
    )


def _local(round_id, clients=CLIENTS, updates=None):
    updates = updates or {}
    return [
        ClientLocalState(
            round_id=round_id,
            client_id=c,
            base_model_hash=HASH,
            update=np.asarray(updates.get(c, [1.0, 0.5, -0.25]), dtype=np.float64),
            num_examples=10,
            num_steps=2,
            loss_history=[1.0, 0.5],
        )
        for c in clients
    ]


def _compression(round_id, clients=CLIENTS, updates=None):
    updates = updates or {}
    return [
        CompressionState(
            round_id=round_id,
            client_id=c,
            uncompressed_hash=HASH,
            update=np.asarray(updates.get(c, [1.0, 0.5, -0.25]), dtype=np.float64),
        )
        for c in clients
    ]


def _aggregation(round_id, aggregate=(0.5, 0.25, -0.125)):
    return AggregationState(
        round_id=round_id,
        received_ids=CLIENTS,
        accepted_ids=CLIENTS,
        rejected_ids=[],
        weights={c: 1.0 / len(CLIENTS) for c in CLIENTS},
        aggregate=np.asarray(aggregate, dtype=np.float64),
    )


def _base_states(round_id, local_clients=CLIENTS):
    return {
        "selection": _selection(round_id),
        "local": _local(round_id, clients=local_clients),
        "compression": _compression(round_id),
        "aggregation": _aggregation(round_id),
    }


def _record_pair(tmp_path, mutate=None, rounds=2, local_clients=CLIENTS, base_mutate=None):
    """Record ref/fail runs identical except ``mutate(states, round_id)`` on fail.

    ``base_mutate`` (applied to BOTH runs first) reshapes the shared baseline.
    """
    for run_id, fn in (("ref", None), ("fail", mutate)):
        recorder = Recorder(tmp_path, run_id)
        for round_id in range(rounds):
            states = _base_states(round_id, local_clients=local_clients)
            if base_mutate is not None:
                base_mutate(states, round_id)
            if fn is not None:
                fn(states, round_id)
            for stage, state in states.items():
                recorder.record(round_id, stage, state)
    return passive_stage_scores(tmp_path, tmp_path, "ref", "fail")


# --------------------------------------------------------------------------
# passive stage-anomaly baseline
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stage,mutate",
    [
        (
            "selection",
            lambda states, r: states.update(
                selection=_selection(r, selected=["c0", "c1"])
            ),
        ),
        (
            "local",
            lambda states, r: states.update(
                local=_local(r, updates={"c1": [5.0, -2.0, 1.0]})
            ),
        ),
        (
            "compression",
            lambda states, r: states.update(
                compression=_compression(r, updates={"c2": [0.0, 0.0, 3.0]})
            ),
        ),
        (
            "aggregation",
            lambda states, r: states.update(
                aggregation=_aggregation(r, aggregate=(4.0, -1.0, 2.0))
            ),
        ),
    ],
    ids=INTERVENABLE_STAGES,
)
def test_single_differing_stage_is_argmax(tmp_path, stage, mutate):
    """Exactly one stage's states differ -> that stage scores highest."""
    scores = _record_pair(tmp_path, mutate)
    assert set(scores) == set(INTERVENABLE_STAGES)
    assert scores[stage] > 0.0
    for other in INTERVENABLE_STAGES:
        if other != stage:
            assert scores[other] == 0.0
    assert passive_localize(scores) == stage


def test_localize_tie_breaks_by_stages_order():
    """Ties resolve deterministically to the earliest stage in STAGES order."""
    assert passive_localize({s: 0.0 for s in INTERVENABLE_STAGES}) == "selection"
    tied = {"selection": 0.1, "local": 0.7, "compression": 0.7, "aggregation": 0.2}
    assert passive_localize(tied) == "local"
    untied = {"selection": 0.1, "local": 0.5, "compression": 0.9, "aggregation": 0.2}
    assert passive_localize(untied) == "compression"


def test_unmatched_clients_count_as_max_deviation(tmp_path):
    """Ref has {c0,c1}, fail has {c0,c2}: union {c0,c1,c2} -> (0 + 1 + 1)/3."""
    def mutate(states, r):
        states["local"] = _local(r, clients=["c0", "c2"])

    scores = _record_pair(tmp_path, mutate, rounds=1, local_clients=["c0", "c1"])
    assert scores["local"] == pytest.approx(2.0 / 3.0)
    assert passive_localize(scores) == "local"


def test_zero_norm_update_guard(tmp_path):
    """0/0 deviation is 0.0; zero vs non-zero is the max deviation 1.0."""
    def zero_base(states, r):
        states["local"] = _local(r, clients=["c0"], updates={"c0": [0.0, 0.0, 0.0]})

    both_zero = _record_pair(
        tmp_path / "same", rounds=1, local_clients=["c0"], base_mutate=zero_base
    )
    assert both_zero["local"] == 0.0

    def nonzero_fail(states, r):
        states["local"] = _local(r, clients=["c0"], updates={"c0": [1.0, 0.0, 0.0]})

    mixed = _record_pair(
        tmp_path / "mixed",
        rounds=1,
        local_clients=["c0"],
        base_mutate=zero_base,
        mutate=nonzero_fail,
    )
    assert mixed["local"] == pytest.approx(1.0)


def test_integration_aggressive_topk_localizes_to_compression(tmp_path):
    """Reference vs T4 aggressive_topk failure run -> compression wins (easy case)."""
    base = dict(
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
    ref_cfg = RunConfig(run_id="t8_ref", failure=None, **base)
    fail_cfg = RunConfig(
        run_id="t8_fail",
        failure=FailureSpecification(
            stage="compression",
            type="aggressive_topk",
            active_rounds=(0, 2),
            severity=1,
            parameters={"k_ratio": 0.04},  # keeps 1 of 22 coordinates
        ),
        **base,
    )
    run(ref_cfg, recorder=Recorder(tmp_path, "t8_ref"), rng=Rng(ref_cfg.seed))
    run(fail_cfg, recorder=Recorder(tmp_path, "t8_fail"), rng=Rng(fail_cfg.seed))

    scores = passive_stage_scores(tmp_path, tmp_path, "t8_ref", "t8_fail")
    assert scores["selection"] == 0.0  # failure never touches selection
    assert passive_localize(scores) == "compression"


# --------------------------------------------------------------------------
# terminal-only baseline
# --------------------------------------------------------------------------

def _record_terminal_run(root, run_id="run"):
    """3 rounds; per-class keys '2'/'10' pin the numeric key ordering."""
    recorder = Recorder(root, run_id)
    accuracies = [0.5, 0.6, 0.9]
    for round_id, accuracy in enumerate(accuracies):
        recorder.record(
            round_id,
            "evaluation",
            OutcomeState(
                round_id=round_id,
                model_hash=HASH,
                metrics={"accuracy": accuracy, "loss": 1.0 - 0.1 * round_id},
                per_class={
                    "10": {"accuracy": 0.1 * (round_id + 1)},
                    "2": {"accuracy": 0.2 * (round_id + 1)},
                },
            ),
        )
        recorder.record(
            round_id,
            "aggregation",
            AggregationState(
                round_id=round_id,
                received_ids=CLIENTS,
                accepted_ids=CLIENTS,
                rejected_ids=[],
                weights={c: 1.0 / len(CLIENTS) for c in CLIENTS},
                aggregate=np.array([3.0, 4.0]) if round_id == 2 else np.ones(2),
            ),
        )


def test_terminal_features_fixed_order_and_stability(tmp_path):
    _record_terminal_run(tmp_path)
    feats = terminal_features(tmp_path, "run")
    assert feats.shape == (6,)
    assert feats.dtype == np.float64
    # [final acc, final loss, class "2" acc, class "10" acc, slope, update norm]
    np.testing.assert_allclose(feats, [0.9, 0.8, 0.6, 0.3, 0.2, 5.0])
    # deterministic across calls
    np.testing.assert_array_equal(feats, terminal_features(tmp_path, "run"))


def test_centroid_classifier_recovers_separated_clusters():
    rng = np.random.default_rng(0)
    centers = {
        "local": np.array([-5.0, 0.0, 1.0, 2.0]),
        "compression": np.array([5.0, 0.0, 1.0, 2.0]),
    }
    X, y = [], []
    for label, center in centers.items():
        for _ in range(5):
            X.append(center + rng.normal(scale=0.1, size=4))
            y.append(label)
    clf = NearestCentroidStageClassifier().fit(X, y)
    for x, label in zip(X, y):
        assert clf.predict(x) == label
    # unseen points near each centroid
    assert clf.predict(centers["local"] + 0.05) == "local"
    assert clf.predict(centers["compression"] - 0.05) == "compression"


def test_centroid_classifier_zero_variance_feature():
    """Constant features get scale 1.0: no NaN/inf, no distance distortion."""
    X = [
        np.array([0.0, 7.0]),
        np.array([0.2, 7.0]),
        np.array([5.0, 7.0]),
        np.array([5.2, 7.0]),
    ]
    y = ["selection", "selection", "aggregation", "aggregation"]
    clf = NearestCentroidStageClassifier().fit(X, y)
    assert clf.predict(np.array([0.1, 7.0])) == "selection"
    assert clf.predict(np.array([5.1, 7.0])) == "aggregation"
    # a shifted constant feature shifts all distances equally
    assert clf.predict(np.array([0.1, 42.0])) == "selection"


def test_centroid_classifier_input_validation():
    clf = NearestCentroidStageClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros(2))
    with pytest.raises(ValueError):
        clf.fit([], [])
    with pytest.raises(ValueError):
        clf.fit([np.zeros(2)], ["local", "compression"])
