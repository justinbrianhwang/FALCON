"""Tier-1 model zoo and the flat-vector <-> torch-model bridge (Task T18).

Torch is confined to this module and ``torch_local.py`` (T18 design decision);
selection/compression/aggregation stay pure numpy on flat arrays. Stage
states carry the flat parameter vector as float32 here (torch-native) — the
recorder hashes dtype+bytes, so per-tier determinism is well-defined; the
synthetic path stays float64.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from .real_data import IMAGE_SHAPES
from .stages import init_params

if TYPE_CHECKING:
    from falcon.replay.rng import Rng  # CONTRACTS §3; deferred to avoid import cycles
    from falcon.schema.config import DatasetConfig, ModelConfig


class SmallCNN(nn.Module):
    """2 conv + 2 fc classifier (Plan §17.2 Tier 1 "small CNN", §18.3).

    Input adapts to the dataset image shape: 28x28x1 (MNIST/FMNIST) or
    32x32x3 (CIFAR/SVHN) — two stride-2 pools reduce the side to ``side // 4``.
    """

    def __init__(self, in_channels: int, side: int, num_classes: int):
        super().__init__()
        if side % 4 != 0:
            raise ValueError(f"image side must be divisible by 4, got {side}")
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        reduced = side // 4
        self.fc1 = nn.Linear(64 * reduced * reduced, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc2(torch.relu(self.fc1(x)))


def _reset_parameters(module: nn.Module, generator: torch.Generator) -> None:
    """Same distributions as ``Conv2d``/``Linear.reset_parameters``, but every
    draw comes from ``generator`` — torch's global RNG is never touched."""
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
    if module.bias is not None:
        # fan-in computation used by torch's own reset_parameters
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def flatten(model: nn.Module) -> np.ndarray:
    """Concatenate all parameters (registration order) into a flat float32 vector."""
    parts = [p.detach().cpu().numpy().reshape(-1) for p in model.parameters()]
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def load_flat(model: nn.Module, vec: np.ndarray) -> nn.Module:
    """Load a flat vector back into ``model``'s parameters (registration order).

    ``vec`` is cast to float32 — the torch-native Tier-1 dtype — before
    loading; the length must match the model exactly.
    """
    flat = np.ascontiguousarray(vec, dtype=np.float32).ravel()
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            if offset + n > flat.shape[0]:
                raise ValueError(
                    f"flat vector too short: need {offset + n}, got {flat.shape[0]}"
                )
            p.copy_(torch.from_numpy(flat[offset : offset + n].reshape(tuple(p.shape))))
            offset += n
    if offset != flat.shape[0]:
        raise ValueError(
            f"flat vector has {flat.shape[0]} entries, model takes {offset}"
        )
    return model


def build_model(cfg: "ModelConfig", dataset: "DatasetConfig", rng: "Rng | None" = None):
    """Build the initial model for ``cfg.name``.

    - ``logistic_regression``: the existing synthetic numpy path — a flat
      float64 init vector from the ``global_init`` stream (requires ``rng``).
    - ``small_cnn``: a ``SmallCNN`` for the dataset's image shape. When ``rng``
      is given, init randomness is a single seed drawn from the
      ``global_init`` stream (CONTRACTS §3) fed to a ``torch.Generator``;
      when ``rng`` is None the module default init is left in place — only
      valid when the weights are immediately overwritten by ``load_flat``
      (torch_local rebuilds), never for a run's initial model.
    """
    if cfg.name == "logistic_regression":
        if rng is None:
            raise ValueError("logistic_regression init requires rng (global_init stream)")
        return init_params(dataset.num_features, dataset.num_classes, rng)
    if cfg.name == "small_cnn":
        if dataset.name not in IMAGE_SHAPES:
            raise ValueError(
                f"small_cnn needs an image dataset (supported: {sorted(IMAGE_SHAPES)}), "
                f"got {dataset.name!r}"
            )
        channels, side = IMAGE_SHAPES[dataset.name]
        model = SmallCNN(channels, side, dataset.num_classes)
        if rng is not None:
            seed = int(rng.stream("global_init").integers(0, 2**31 - 1))
            generator = torch.Generator().manual_seed(seed)
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    _reset_parameters(module, generator)
        return model
    raise ValueError(f"unknown model {cfg.name!r}")
