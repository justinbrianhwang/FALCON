"""A1 — aggregation-stage failure: the server uses wrong sample-count weights."""
from __future__ import annotations

import math

from ..base import FailureInjector

_MODES = ("uniform", "swapped", "corrupted")


class WrongSampleWeightsInjector(FailureInjector):
    """Replace the aggregation weights while active.

    Parameters (``spec.parameters``):
      - ``mode``:
          - ``"uniform"``: every client gets the same weight (1.0);
          - ``"swapped"``: the weight values are reversed across sorted
            client ids (smallest id gets the largest id's weight, ...);
          - ``"corrupted"``: each weight is multiplied by a factor drawn from
            stream ``failure.aggregation``, log-uniform in
            ``[10**-intensity, 10**intensity)`` (``10 ** uniform(-i, i)``),
            one draw per client per active round in sorted-id order.
      - ``intensity`` (``"corrupted"`` only, Task T15): spread of the
        log-uniform factors, in (0, 4]; default 1.0 reproduces the original
        fixed [0.1, 10) range bit-for-bit (same stream, same draws).
        ``"uniform"``/``"swapped"`` reject an ``intensity`` parameter.

    ``aggregate`` re-normalizes the weights as usual, so only their relative
    values matter.
    """

    stage = "aggregation"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        self._mode = str(spec.parameters["mode"])
        if self._mode not in _MODES:
            raise ValueError(
                f"wrong_sample_weights mode must be one of {_MODES}, "
                f"got {self._mode!r}"
            )
        if self._mode == "corrupted":
            self._intensity = float(spec.parameters.get("intensity", 1.0))
            if not math.isfinite(self._intensity) or not 0.0 < self._intensity <= 4.0:
                raise ValueError(
                    f"intensity must be in (0, 4], got {self._intensity}"
                )
        elif "intensity" in spec.parameters:
            raise ValueError(
                f"intensity is only valid for mode 'corrupted', "
                f"got mode {self._mode!r}"
            )

    @property
    def mode(self) -> str:
        return self._mode

    def weights(self, weights: dict[str, float], round_id: int) -> dict[str, float]:
        if not self.active(round_id):
            return dict(weights)
        ids = sorted(weights)
        if self._mode == "uniform":
            return {cid: 1.0 for cid in ids}
        if self._mode == "swapped":
            values = [weights[cid] for cid in ids]
            values.reverse()
            return dict(zip(ids, values))
        gen = self._gen()
        intensity = self._intensity
        return {
            cid: weights[cid] * float(10.0 ** gen.uniform(-intensity, intensity))
            for cid in ids
        }
