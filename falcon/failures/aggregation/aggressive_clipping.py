"""A2 - aggregation-stage failure: over-aggressive update clipping."""
from __future__ import annotations

from ..base import FailureInjector


class AggressiveClippingInjector(FailureInjector):
    """Add a severity-scaled clip norm to the aggregation config."""

    stage = "aggregation"
    _CLIP_NORMS = {1: 1.0, 2: 0.1, 3: 0.01}

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        try:
            self._clip_norm = self._CLIP_NORMS[spec.severity]
        except KeyError:
            raise ValueError(
                f"aggressive_clipping severity must be 1, 2, or 3, got {spec.severity}"
            ) from None

    def aggregation_cfg(self, cfg, round_id):
        if not self.active(round_id):
            return cfg.model_copy()
        return cfg.model_copy(
            update={"parameters": {**cfg.parameters, "clip_norm": self._clip_norm}}
        )
