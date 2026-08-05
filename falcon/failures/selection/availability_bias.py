"""S2 - selection-stage failure: deterministic clients have low availability."""
from __future__ import annotations

import math

from ..base import FailureInjector


class AvailabilityBiasInjector(FailureInjector):
    """Stochastically exclude a sorted prefix of clients while active."""

    stage = "selection"
    _AVAILABILITY = {1: 0.7, 2: 0.4, 3: 0.1}

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        try:
            default_availability = self._AVAILABILITY[spec.severity]
        except KeyError:
            raise ValueError(
                f"availability_bias severity must be 1, 2, or 3, got {spec.severity}"
            ) from None
        self._fraction = float(spec.parameters["biased_fraction"])
        self._availability = float(
            spec.parameters.get("availability", default_availability)
        )
        if not math.isfinite(self._fraction) or not 0.0 <= self._fraction <= 1.0:
            raise ValueError(
                f"biased_fraction must be in [0, 1], got {self._fraction}"
            )
        if (
            not math.isfinite(self._availability)
            or not 0.0 <= self._availability <= 1.0
        ):
            raise ValueError(
                f"availability must be in [0, 1], got {self._availability}"
            )
        ids = sorted(partition)
        self._biased = frozenset(ids[: math.ceil(self._fraction * len(ids))])

    @property
    def biased_clients(self) -> frozenset[str]:
        return self._biased

    def candidate_pool(self, pool: list[str], round_id: int) -> list[str]:
        if not self.active(round_id):
            return list(pool)
        return [
            cid
            for cid in sorted(pool)
            if cid not in self._biased
            or self._rng.stream(
                f"failure.selection.{cid}.round.{round_id}"
            ).random()
            < self._availability
        ]
