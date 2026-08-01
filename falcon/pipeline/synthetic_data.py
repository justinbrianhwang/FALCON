"""Synthetic logistic-regression data: Gaussian class clusters (Task T2).

The partition depends ONLY on ``cfg.seed`` (its own ``np.random.Generator``),
never on the run seed, so the same ``DatasetConfig`` always yields the same
client datasets regardless of the surrounding run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from falcon.schema import DatasetConfig

_CLASS_SEP = 1.0  # scale of the cluster centers (class means)
_NOISE = 1.0  # within-cluster noise scale
_MINORITY_CONCENTRATION = 0.9  # fraction of minority-class samples on minority clients
_EVAL_SEED_OFFSET = 7919  # fixed offset deriving the eval-set seed from cfg.seed


@dataclass
class ClientData:
    x: np.ndarray  # (n, d) float64 features
    y: np.ndarray  # (n,) int64 class labels


EvalData = ClientData


def _cluster_centers(gen: np.random.Generator, num_classes: int, num_features: int) -> np.ndarray:
    return gen.normal(0.0, _CLASS_SEP, size=(num_classes, num_features)).astype(np.float64)


def _sample(
    gen: np.random.Generator,
    centers: np.ndarray,
    labels: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    noise = gen.normal(0.0, _NOISE, size=(labels.shape[0], centers.shape[1]))
    return (centers[labels] + shift + noise).astype(np.float64)


def make_partition(cfg: DatasetConfig) -> dict[str, ClientData]:
    """Partition synthetic Gaussian class clusters across ``cfg.num_clients``.

    - ``heterogeneity`` scales a per-client shift of the feature means
      (0.0 = IID).
    - ``minority_class`` / ``minority_client_fraction``: that fraction of
      clients draws ~90 % of its samples from ``minority_class``.
    """
    gen = np.random.default_rng(cfg.seed)  # partition seed, independent of run seed
    centers = _cluster_centers(gen, cfg.num_classes, cfg.num_features)

    minority_clients: set[int] = set()
    if cfg.minority_class is not None:
        n_minority = int(round(cfg.num_clients * cfg.minority_client_fraction))
        minority_clients = set(gen.permutation(cfg.num_clients)[:n_minority].tolist())

    partition: dict[str, ClientData] = {}
    for i in range(cfg.num_clients):
        n = cfg.samples_per_client
        if cfg.heterogeneity > 0.0:
            shift = gen.normal(0.0, cfg.heterogeneity, size=cfg.num_features)
        else:
            shift = np.zeros(cfg.num_features, dtype=np.float64)
        if i in minority_clients:
            is_minority = gen.random(n) < _MINORITY_CONCENTRATION
            labels = np.where(
                is_minority, cfg.minority_class, gen.integers(0, cfg.num_classes, size=n)
            ).astype(np.int64)
        else:
            labels = gen.integers(0, cfg.num_classes, size=n, dtype=np.int64)
        x = _sample(gen, centers, labels, shift)
        partition[f"client_{i}"] = ClientData(x=x, y=labels)
    return partition


def make_eval_data(cfg: DatasetConfig, num_samples: int = 500) -> EvalData:
    """Global eval set: same cluster centers as the partition, no client shift.

    Uses a fixed derived seed (``cfg.seed + offset``) so it is identical for
    every run built on the same ``DatasetConfig``.
    """
    centers = _cluster_centers(
        np.random.default_rng(cfg.seed), cfg.num_classes, cfg.num_features
    )
    gen = np.random.default_rng(cfg.seed + _EVAL_SEED_OFFSET)
    labels = gen.integers(0, cfg.num_classes, size=num_samples, dtype=np.int64)
    x = _sample(gen, centers, labels, np.zeros(cfg.num_features, dtype=np.float64))
    return EvalData(x=x, y=labels)
