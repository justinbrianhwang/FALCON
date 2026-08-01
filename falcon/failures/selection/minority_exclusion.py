"""S1 — selection-stage failure: minority-class clients are dropped from the pool."""
from __future__ import annotations

import numpy as np

from ..base import FailureInjector


class MinorityExclusionInjector(FailureInjector):
    """Drop minority-heavy clients from the candidate pool while active.

    Parameters (``spec.parameters``):
      - ``target_class``: class label whose carriers are excluded.
      - ``exclusion_probability``: independent per-client drop probability,
        drawn from stream ``failure.selection`` once per active round per
        minority-heavy client (in pool order).

    A client is "minority-heavy" when its within-client share of
    ``target_class`` samples exceeds the dataset-wide share of that class
    (i.e. the uniform share the class would hold if its samples were spread
    evenly across the whole partition). Computed once, deterministically,
    from ``partition`` at construction.
    """

    stage = "selection"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        params = spec.parameters
        self._target_class = int(params["target_class"])
        self._exclusion_probability = float(params["exclusion_probability"])
        if not 0.0 <= self._exclusion_probability <= 1.0:
            raise ValueError(
                f"exclusion_probability must be in [0, 1], "
                f"got {self._exclusion_probability}"
            )
        self._minority_heavy = self._find_minority_heavy(partition)

    def _find_minority_heavy(self, partition) -> frozenset[str]:
        counts = {
            cid: int((data.y == self._target_class).sum())
            for cid, data in partition.items()
        }
        total_target = sum(counts.values())
        total = sum(data.y.shape[0] for data in partition.values())
        if total_target == 0 or total == 0:
            return frozenset()
        uniform_share = total_target / total
        return frozenset(
            cid
            for cid, data in partition.items()
            if counts[cid] / data.y.shape[0] > uniform_share
        )

    @property
    def minority_heavy_clients(self) -> frozenset[str]:
        return self._minority_heavy

    def candidate_pool(self, pool: list[str], round_id: int) -> list[str]:
        if not self.active(round_id):
            return list(pool)
        gen = self._gen()
        return [
            cid
            for cid in pool
            if cid not in self._minority_heavy
            or gen.random() >= self._exclusion_probability
        ]
