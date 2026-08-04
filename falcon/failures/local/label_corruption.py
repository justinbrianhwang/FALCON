"""L4 - local-training failure: corrupt labels for deterministic clients."""
from __future__ import annotations

import math

import numpy as np

from falcon.pipeline.synthetic_data import ClientData

from ..base import FailureInjector


class LabelCorruptionInjector(FailureInjector):
    """Randomly replace labels with uniformly sampled other classes."""

    stage = "local"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        self._fraction = float(spec.parameters["fraction_clients"])
        self._probability = float(spec.parameters["flip_probability"])
        if not math.isfinite(self._fraction) or not 0.0 <= self._fraction <= 1.0:
            raise ValueError(
                f"fraction_clients must be in [0, 1], got {self._fraction}"
            )
        if not math.isfinite(self._probability) or not 0.0 <= self._probability <= 1.0:
            raise ValueError(
                f"flip_probability must be in [0, 1], got {self._probability}"
            )
        ids = sorted(partition)
        self._affected = frozenset(ids[: math.ceil(self._fraction * len(ids))])
        labels = [data.y for data in partition.values() if data.y.size]
        self._num_classes = max(int(label.max()) for label in labels) + 1 if labels else 0

    @property
    def affected_clients(self) -> frozenset[str]:
        return self._affected

    def local_data(self, client_id, data, round_id):
        if not self.active(round_id) or client_id not in self._affected:
            return data
        if self._num_classes < 2:
            return data
        gen = self._rng.stream(f"failure.local.{client_id}.round.{round_id}")
        flipped = np.array(data.y, copy=True)
        mask = gen.random(flipped.shape[0]) < self._probability
        offsets = gen.integers(1, self._num_classes, size=int(mask.sum()), dtype=flipped.dtype)
        flipped[mask] = (flipped[mask] + offsets) % self._num_classes
        return ClientData(x=data.x, y=flipped)
