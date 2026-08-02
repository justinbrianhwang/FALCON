"""Real image datasets from processed pickles (Task T18, Plan §17.2 Tier 1).

Loads ONLY the standardized pickles at ``falcon.data_paths.processed_path(name)``
written by ``scripts/prepare_data.py`` — no torchvision import anywhere in the
pipeline. ``ClientData.x`` is float32 NCHW normalized to [0, 1] (``x / 255``);
``y`` is int64. (The synthetic path keeps its float64 ``(n, d)`` features; the
container is shared, the dtype is per-tier — see CONTRACTS-adjacent note in
``falcon/schema/states.py``.)

Like the synthetic partition, the real partition depends ONLY on
``DatasetConfig.seed`` (its own ``np.random.Generator``), never on the run
seed, so the same ``DatasetConfig`` always yields the same client datasets.
``dirichlet_alpha=None`` means IID; otherwise each class is split across
clients by ``Dirichlet(alpha)`` proportions (Plan §18.2). Eval is the FULL
test split of the dataset.
"""
from __future__ import annotations

import pickle

import numpy as np

from falcon.data_paths import processed_path
from falcon.schema import DatasetConfig

from .synthetic_data import ClientData, EvalData

# dataset -> (channels, side); pickles store uint8 images, grayscale (n, s, s)
# or color NHWC (n, s, s, c). SVHN/CIFAR are 32x32x3, MNIST/FMNIST 28x28x1.
IMAGE_SHAPES: dict[str, tuple[int, int]] = {
    "mnist": (1, 28),
    "fmnist": (1, 28),
    "cifar10": (3, 32),
    "cifar100": (3, 32),
    "svhn": (3, 32),
}


def _load_pickle(name: str) -> dict:
    path = processed_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"processed dataset not found: {path} — run "
            f"`python scripts/prepare_data.py --datasets {name}` first"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _to_nchw(x: np.ndarray, name: str) -> np.ndarray:
    """uint8 pickle images -> contiguous float32 NCHW in [0, 1]."""
    channels, side = IMAGE_SHAPES[name]
    if x.ndim == 3:  # grayscale (n, side, side)
        x = x[:, None, :, :]
    elif x.ndim == 4:  # color NHWC (n, side, side, channels)
        x = np.transpose(x, (0, 3, 1, 2))
    else:
        raise ValueError(f"{name}: expected 3- or 4-d images, got shape {x.shape}")
    if tuple(x.shape[1:]) != (channels, side, side):
        raise ValueError(
            f"{name}: expected image shape {(channels, side, side)}, got {tuple(x.shape[1:])}"
        )
    # python-float division keeps float32 (NEP 50 weak scalars)
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0


def _check_labels(y: np.ndarray, cfg: DatasetConfig) -> None:
    num_classes = int(y.max()) + 1 if y.shape[0] else 0
    if y.min() < 0 or num_classes != cfg.num_classes:
        raise ValueError(
            f"{cfg.name}: pickle has {num_classes} classes, "
            f"DatasetConfig.num_classes={cfg.num_classes}"
        )


def _iid_split(
    candidate_idx: np.ndarray, num_clients: int, gen: np.random.Generator
) -> list[list[np.ndarray]]:
    """Even split of a shuffled candidate pool over all clients (Plan §18.2 IID)."""
    perm = candidate_idx[gen.permutation(candidate_idx.shape[0])]
    return [[part] for part in np.array_split(perm, num_clients)]


def _dirichlet_split(
    y: np.ndarray,
    num_classes: int,
    candidate_idx: np.ndarray,
    num_clients: int,
    alpha: float,
    gen: np.random.Generator,
) -> list[list[np.ndarray]]:
    """Per class, deal samples by ``Dirichlet(alpha)`` proportions (Plan §18.2).

    Draw order is fixed (class order; permutation before proportions) so the
    partition is a pure function of ``gen``'s seed.
    """
    buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        cls_idx = candidate_idx[y[candidate_idx] == c]
        cls_idx = cls_idx[gen.permutation(cls_idx.shape[0])]
        proportions = gen.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions)[:-1] * cls_idx.shape[0]).astype(np.int64)
        for i, part in enumerate(np.split(cls_idx, cuts)):
            if part.shape[0]:
                buckets[i].append(part)
    return buckets


def _partition_indices(
    y: np.ndarray, cfg: DatasetConfig, gen: np.random.Generator
) -> list[np.ndarray]:
    """Assign every train index to exactly one client.

    ``minority_class`` / ``minority_client_fraction`` are honored like the
    synthetic path (synthetic_data.make_partition): a designated subset of
    ``max(1, round(num_clients * fraction))`` clients — drawn FIRST from the
    partition generator, as in synthetic — holds ALL samples of the minority
    class (real data cannot synthesize the thin residual spread the synthetic
    generator plants on other clients, so concentration is 1.0). Every other
    class is partitioned IID or Dirichlet over ALL clients, so designated
    clients become minority-heavy (within-client share above the dataset-wide
    share) and are found by the selection-failure targeting exactly as in the
    synthetic tier.
    """
    all_idx = np.arange(y.shape[0])
    designated: list[int] | None = None
    rest = all_idx
    minority_idx: np.ndarray | None = None
    if cfg.minority_class is not None:
        if not 0 <= cfg.minority_class < cfg.num_classes:
            raise ValueError(
                f"minority_class {cfg.minority_class} out of range for "
                f"{cfg.num_classes} classes"
            )
        n_designated = max(1, int(round(cfg.num_clients * cfg.minority_client_fraction)))
        designated = sorted(gen.permutation(cfg.num_clients)[:n_designated].tolist())
        minority_idx = all_idx[y == cfg.minority_class]
        rest = all_idx[y != cfg.minority_class]

    if cfg.dirichlet_alpha is None:
        buckets = _iid_split(rest, cfg.num_clients, gen)
    else:
        if not np.isfinite(cfg.dirichlet_alpha) or cfg.dirichlet_alpha <= 0.0:
            raise ValueError(f"dirichlet_alpha must be positive, got {cfg.dirichlet_alpha}")
        buckets = _dirichlet_split(
            y, cfg.num_classes, rest, cfg.num_clients, cfg.dirichlet_alpha, gen
        )

    if minority_idx is not None and designated is not None:
        shuffled = minority_idx[gen.permutation(minority_idx.shape[0])]
        for cid, part in zip(designated, np.array_split(shuffled, len(designated))):
            buckets[cid].append(part)

    return [
        np.sort(np.concatenate(bucket)) if bucket else np.empty(0, dtype=np.int64)
        for bucket in buckets
    ]


def load_partition(cfg: DatasetConfig) -> dict[str, ClientData]:
    """Partition the processed train split across ``cfg.num_clients`` clients."""
    if cfg.name not in IMAGE_SHAPES:
        raise ValueError(
            f"load_partition: dataset {cfg.name!r} is not a real image dataset "
            f"(supported: {sorted(IMAGE_SHAPES)}); use synthetic_data.make_partition"
        )
    raw = _load_pickle(cfg.name)
    x = _to_nchw(raw["x_train"], cfg.name)
    y = np.asarray(raw["y_train"], dtype=np.int64)
    _check_labels(y, cfg)
    gen = np.random.default_rng(cfg.seed)  # partition seed, independent of run seed
    indices = _partition_indices(y, cfg, gen)
    return {
        f"client_{i}": ClientData(x=x[idx], y=y[idx])
        for i, idx in enumerate(indices)
    }


def load_eval_data(cfg: DatasetConfig) -> EvalData:
    """Global eval set: the FULL test split of the processed pickle."""
    if cfg.name not in IMAGE_SHAPES:
        raise ValueError(
            f"load_eval_data: dataset {cfg.name!r} is not a real image dataset "
            f"(supported: {sorted(IMAGE_SHAPES)})"
        )
    raw = _load_pickle(cfg.name)
    x = _to_nchw(raw["x_test"], cfg.name)
    y = np.asarray(raw["y_test"], dtype=np.int64)
    _check_labels(y, cfg)
    return EvalData(x=x, y=y)
