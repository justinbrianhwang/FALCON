"""A1 — aggregation-stage failure: the server uses wrong sample-count weights."""
from __future__ import annotations

import math

from ..base import FailureInjector
from ..targeting import minority_heavy_clients

_MODES = ("uniform", "swapped", "corrupted", "biased")


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
            one draw per client per active round in sorted-id order;
          - ``"biased"`` (Task T17): minority-heavy clients — within-client
            share of ``target_class`` above the dataset-wide share, the same
            targeting rule as selection's ``minority_exclusion`` — get their
            weight multiplied by ``weight_multiplier`` BEFORE normalization.
            Fully deterministic: no RNG at all, and ``weight_multiplier=1.0``
            is an exact no-op (``x * 1.0 == x`` in IEEE-754), so lower
            multipliers are more severe, deterministically monotone.
      - ``intensity`` (``"corrupted"`` only, Task T15): spread of the
        log-uniform factors, in (0, 4]; default 1.0 reproduces the original
        fixed [0.1, 10) range bit-for-bit (same stream, same draws).
      - ``weight_multiplier`` (``"biased"`` only, required): down-weighting
        factor for minority-heavy clients, in (0, 1].
      - ``target_class`` (``"biased"`` only, required): class label defining
        the minority-heavy targeting rule.

    Any knob outside its owning mode is rejected. ``aggregate`` re-normalizes
    the weights as usual, so only their relative values matter.
    """

    stage = "aggregation"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        params = spec.parameters
        self._mode = str(params["mode"])
        if self._mode not in _MODES:
            raise ValueError(
                f"wrong_sample_weights mode must be one of {_MODES}, "
                f"got {self._mode!r}"
            )
        if self._mode == "corrupted":
            self._reject_foreign(params, ("weight_multiplier", "target_class"), "'biased'")
            self._intensity = float(params.get("intensity", 1.0))
            if not math.isfinite(self._intensity) or not 0.0 < self._intensity <= 4.0:
                raise ValueError(
                    f"intensity must be in (0, 4], got {self._intensity}"
                )
        elif self._mode == "biased":
            self._reject_foreign(params, ("intensity",), "'corrupted'")
            if "weight_multiplier" not in params:
                raise ValueError("weight_multiplier is required for mode 'biased'")
            self._weight_multiplier = float(params["weight_multiplier"])
            if (
                not math.isfinite(self._weight_multiplier)
                or not 0.0 < self._weight_multiplier <= 1.0
            ):
                raise ValueError(
                    f"weight_multiplier must be in (0, 1], "
                    f"got {self._weight_multiplier}"
                )
            if "target_class" not in params:
                raise ValueError("target_class is required for mode 'biased'")
            self._target_class = int(params["target_class"])
            self._minority_heavy = minority_heavy_clients(partition, self._target_class)
        else:
            self._reject_foreign(params, ("intensity",), "'corrupted'")
            self._reject_foreign(params, ("weight_multiplier", "target_class"), "'biased'")

    def _reject_foreign(self, params, knobs: tuple[str, ...], owner: str) -> None:
        for knob in knobs:
            if knob in params:
                raise ValueError(
                    f"{knob} is only valid for mode {owner}, "
                    f"got mode {self._mode!r}"
                )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def minority_heavy_clients(self) -> frozenset[str]:
        """Clients down-weighted in ``"biased"`` mode (empty otherwise)."""
        return getattr(self, "_minority_heavy", frozenset())

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
        if self._mode == "biased":
            multiplier = self._weight_multiplier
            targeted = self._minority_heavy
            return {
                cid: weights[cid] * multiplier if cid in targeted else weights[cid]
                for cid in ids
            }
        gen = self._gen()
        intensity = self._intensity
        return {
            cid: weights[cid] * float(10.0 ** gen.uniform(-intensity, intensity))
            for cid in ids
        }
