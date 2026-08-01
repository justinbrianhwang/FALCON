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
_MINORITY_PREVALENCE = 0.1  # intended global share of minority-class samples
_MINORITY_CONCENTRATION = 0.9  # share of ALL minority-class samples on designated clients
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


def _distribute(total: int, capacities: list[int]) -> list[int]:
    """Spread ``total`` items across bins as evenly as possible, capped per bin."""
    counts = [0] * len(capacities)
    remaining = min(total, sum(capacities))
    while remaining > 0:
        active = [i for i, cap in enumerate(capacities) if counts[i] < cap]
        if not active:
            break
        share, extra = divmod(remaining, len(active))
        for j, i in enumerate(active):
            give = min(share + (1 if j < extra else 0), capacities[i] - counts[i])
            counts[i] += give
            remaining -= give
    return counts


def _labels_without(
    gen: np.random.Generator, n: int, num_classes: int, excluded: int
) -> np.ndarray:
    """Uniform labels over every class except ``excluded`` (minority suppression)."""
    allowed = [c for c in range(num_classes) if c != excluded]
    if not allowed:  # degenerate single-class dataset: nothing left to suppress
        return gen.integers(0, num_classes, size=n, dtype=np.int64)
    return gen.choice(allowed, size=n).astype(np.int64)


def make_partition(cfg: DatasetConfig) -> dict[str, ClientData]:
    """Partition synthetic Gaussian class clusters across ``cfg.num_clients``.

    - ``heterogeneity`` scales a per-client shift of the feature means
      (0.0 = IID).
    - ``minority_class`` / ``minority_client_fraction``: a designated subset of
      ``max(1, round(num_clients * fraction))`` clients holds
      ~``_MINORITY_CONCENTRATION`` of ALL minority-class samples, while the
      class stays globally rare (~``_MINORITY_PREVALENCE`` of all samples).
      The remaining ~(1 - concentration) share is spread thinly over the
      non-designated clients, suppressing the minority label far below the
      uniform rate there. If the designated clients cannot hold the requested
      share, they are filled to capacity and the surplus spills over.
    """
    gen = np.random.default_rng(cfg.seed)  # partition seed, independent of run seed
    centers = _cluster_centers(gen, cfg.num_classes, cfg.num_features)

    minority_counts: dict[int, int] = {}
    if cfg.minority_class is not None:
        if not 0 <= cfg.minority_class < cfg.num_classes:
            raise ValueError(
                f"minority_class {cfg.minority_class} out of range for "
                f"{cfg.num_classes} classes"
            )
        n_designated = max(1, int(round(cfg.num_clients * cfg.minority_client_fraction)))
        designated = set(gen.permutation(cfg.num_clients)[:n_designated].tolist())
        others = sorted(set(range(cfg.num_clients)) - designated)
        total_samples = cfg.num_clients * cfg.samples_per_client
        n_minority = int(round(total_samples * _MINORITY_PREVALENCE))
        n_on_designated = int(round(n_minority * _MINORITY_CONCENTRATION))
        designated_counts = _distribute(
            n_on_designated, [cfg.samples_per_client] * n_designated
        )
        minority_counts = dict(zip(sorted(designated), designated_counts))
        other_counts = _distribute(
            n_minority - sum(designated_counts),
            [cfg.samples_per_client] * len(others),
        )
        minority_counts.update(zip(others, other_counts))

    partition: dict[str, ClientData] = {}
    for i in range(cfg.num_clients):
        n = cfg.samples_per_client
        if cfg.heterogeneity > 0.0:
            shift = gen.normal(0.0, cfg.heterogeneity, size=cfg.num_features)
        else:
            shift = np.zeros(cfg.num_features, dtype=np.float64)
        if cfg.minority_class is None:
            labels = gen.integers(0, cfg.num_classes, size=n, dtype=np.int64)
        else:
            n_minority = minority_counts.get(i, 0)
            labels = np.concatenate(
                [
                    np.full(n_minority, cfg.minority_class, dtype=np.int64),
                    _labels_without(gen, n - n_minority, cfg.num_classes, cfg.minority_class),
                ]
            )
            labels = labels[gen.permutation(n)]  # shuffle the minority block away
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
