"""Invariant tests for the synthetic partition (T2-F finding 2).

Documented minority semantics: a designated subset of
``max(1, round(num_clients * minority_client_fraction))`` clients holds
~``_MINORITY_CONCENTRATION`` of ALL minority-class samples; the class stays
globally rare (~``_MINORITY_PREVALENCE``) and is suppressed elsewhere.
"""
import numpy as np
import pytest

from falcon.pipeline.synthetic_data import (
    _MINORITY_CONCENTRATION,
    _MINORITY_PREVALENCE,
    make_eval_data,
    make_partition,
)
from falcon.schema import DatasetConfig


def _minority_cfg(**overrides) -> DatasetConfig:
    base = dict(
        num_clients=10,
        num_features=5,
        num_classes=2,
        samples_per_client=200,
        minority_class=1,
        minority_client_fraction=0.2,
        seed=123,
    )
    return DatasetConfig(**(base | overrides))


def _minority_counts(partition: dict, minority_class: int) -> dict[str, int]:
    return {
        cid: int((data.y == minority_class).sum()) for cid, data in partition.items()
    }


def _designated_clients(
    partition: dict, minority_class: int, n_designated: int
) -> set[str]:
    """The designated subset = the clients with the highest minority counts."""
    counts = _minority_counts(partition, minority_class)
    ranked = sorted(counts, key=lambda cid: (-counts[cid], cid))
    return set(ranked[:n_designated])


def test_minority_class_is_globally_rare():
    cfg = _minority_cfg()
    partition = make_partition(cfg)
    total = sum(data.y.shape[0] for data in partition.values())
    minority = sum(_minority_counts(partition, cfg.minority_class).values())
    assert total == cfg.num_clients * cfg.samples_per_client
    assert minority / total == pytest.approx(_MINORITY_PREVALENCE, abs=1.0 / total)


def test_minority_samples_concentrate_on_designated_subset():
    cfg = _minority_cfg()
    partition = make_partition(cfg)
    n_designated = max(1, round(cfg.num_clients * cfg.minority_client_fraction))
    designated = _designated_clients(partition, cfg.minority_class, n_designated)
    counts = _minority_counts(partition, cfg.minority_class)
    share = sum(counts[cid] for cid in designated) / sum(counts.values())
    assert share == pytest.approx(
        _MINORITY_CONCENTRATION, abs=1.0 / sum(counts.values())
    )


def test_minority_label_suppressed_on_non_designated_clients():
    cfg = _minority_cfg()
    partition = make_partition(cfg)
    n_designated = max(1, round(cfg.num_clients * cfg.minority_client_fraction))
    designated = _designated_clients(partition, cfg.minority_class, n_designated)
    counts = _minority_counts(partition, cfg.minority_class)
    # a clear gap separates the designated subset from the suppressed rest
    assert min(counts[cid] for cid in designated) > max(
        counts[cid] for cid in counts.keys() - designated
    )
    for cid in counts.keys() - designated:
        rate = counts[cid] / cfg.samples_per_client
        assert rate < 1.0 / cfg.num_classes  # well below the uniform rate


def test_designated_subset_never_empty():
    # round(2 * 0.2) == 0 must still yield one designated client.
    cfg = _minority_cfg(num_clients=2, samples_per_client=100, minority_client_fraction=0.2)
    partition = make_partition(cfg)
    counts = _minority_counts(partition, cfg.minority_class)
    assert sum(counts.values()) > 0
    designated = _designated_clients(partition, cfg.minority_class, 1)
    share = sum(counts[cid] for cid in designated) / sum(counts.values())
    assert share == pytest.approx(_MINORITY_CONCENTRATION, abs=0.05)


def test_partition_deterministic_from_dataset_seed():
    cfg = _minority_cfg()
    a = make_partition(cfg)
    b = make_partition(cfg)
    assert a.keys() == b.keys()
    for cid in a:
        np.testing.assert_array_equal(a[cid].x, b[cid].x)
        np.testing.assert_array_equal(a[cid].y, b[cid].y)

    other = make_partition(_minority_cfg(seed=124))
    assert any(
        not np.array_equal(a[cid].y, other[cid].y) for cid in a
    ), "different dataset seeds must produce different partitions"


def test_partition_dtypes_and_shapes():
    cfg = _minority_cfg(heterogeneity=0.5)
    partition = make_partition(cfg)
    assert sorted(partition) == [f"client_{i}" for i in range(cfg.num_clients)]
    for data in partition.values():
        assert data.x.dtype == np.float64
        assert data.y.dtype == np.int64
        assert data.x.shape == (cfg.samples_per_client, cfg.num_features)
        assert data.y.shape == (cfg.samples_per_client,)
        assert set(np.unique(data.y)) <= set(range(cfg.num_classes))


def test_no_minority_class_keeps_uniform_labels():
    cfg = _minority_cfg(minority_class=None)
    partition = make_partition(cfg)
    all_labels = np.concatenate([data.y for data in partition.values()])
    assert set(np.unique(all_labels)) == {0, 1}
    assert (all_labels == 1).mean() == pytest.approx(0.5, abs=0.05)


def test_minority_class_out_of_range_rejected():
    with pytest.raises(ValueError, match="minority_class"):
        make_partition(_minority_cfg(minority_class=5))


def test_eval_data_deterministic_and_float64():
    cfg = _minority_cfg()
    a = make_eval_data(cfg)
    b = make_eval_data(cfg)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.x.dtype == np.float64
    assert a.y.dtype == np.int64
    assert a.x.shape[1] == cfg.num_features
