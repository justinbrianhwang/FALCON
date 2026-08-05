"""L5 - local-training failure: sign-flip and scale selected updates."""
from __future__ import annotations

import math

from ..base import FailureInjector


class ModelPoisoningInjector(FailureInjector):
    """Replace affected clients' true updates with ``-scale * update``."""

    stage = "local"
    _SCALES = {1: 1.0, 2: 5.0, 3: 20.0}

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        try:
            default_scale = self._SCALES[spec.severity]
        except KeyError:
            raise ValueError(
                f"model_poisoning severity must be 1, 2, or 3, got {spec.severity}"
            ) from None
        self._fraction = float(spec.parameters["fraction_clients"])
        self._scale = float(spec.parameters.get("scale", default_scale))
        if not math.isfinite(self._fraction) or not 0.0 <= self._fraction <= 1.0:
            raise ValueError(
                f"fraction_clients must be in [0, 1], got {self._fraction}"
            )
        if not math.isfinite(self._scale) or self._scale <= 0.0:
            raise ValueError(f"scale must be positive and finite, got {self._scale}")
        ids = sorted(partition)
        self._affected = frozenset(ids[: math.ceil(self._fraction * len(ids))])

    @property
    def affected_clients(self) -> frozenset[str]:
        return self._affected

    def local_state(self, client_id, state, round_id):
        if not self.active(round_id) or client_id not in self._affected:
            return state
        return state.model_copy(update={"update": -self._scale * state.update})
