"""C2 - compression-stage failure: low-bit uniform quantization."""
from __future__ import annotations

from ..base import FailureInjector


class AggressiveQuantizationInjector(FailureInjector):
    """Replace compression with severity-scaled quantization."""

    stage = "compression"
    _BITS = {1: 8, 2: 4, 3: 2}

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        try:
            self._bits = self._BITS[spec.severity]
        except KeyError:
            raise ValueError(
                f"aggressive_quantization severity must be 1, 2, or 3, got {spec.severity}"
            ) from None

    def compression_cfg(self, client_id, cfg, round_id):
        if not self.active(round_id):
            return cfg.model_copy()
        return cfg.model_copy(
            update={"kind": "quantization", "parameters": {"bits": self._bits}}
        )
