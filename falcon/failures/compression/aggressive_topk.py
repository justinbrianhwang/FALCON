"""C1 — compression-stage failure: updates are aggressively top-k sparsified."""
from __future__ import annotations

import math

from ..base import FailureInjector


class AggressiveTopKInjector(FailureInjector):
    """Swap the compression config to ``topk`` with an aggressive ratio.

    Parameters (``spec.parameters``):
      - ``k_ratio``: fraction of coordinates kept by magnitude, in (0, 1];
        ``stages.compress`` keeps ``ceil(k_ratio * n)`` of them.
      - ``affected_clients``: optional id list; defaults to every client in
        the partition.

    Uses no randomness — top-k itself is exact and deterministic.
    """

    stage = "compression"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        params = spec.parameters
        self._k_ratio = float(params["k_ratio"])
        if not math.isfinite(self._k_ratio) or not 0.0 < self._k_ratio <= 1.0:
            raise ValueError(f"k_ratio must be in (0, 1], got {self._k_ratio}")
        affected = params.get("affected_clients")
        if affected is None:
            self._affected = frozenset(partition)
        else:
            unknown = sorted(set(affected) - set(partition))
            if unknown:
                raise ValueError(f"affected_clients not in partition: {unknown}")
            self._affected = frozenset(str(cid) for cid in affected)

    @property
    def affected_clients(self) -> frozenset[str]:
        return self._affected

    def compression_cfg(self, client_id, cfg, round_id):
        if not self.active(round_id) or client_id not in self._affected:
            return cfg.model_copy()
        return cfg.model_copy(
            update={"kind": "topk", "parameters": {"k_ratio": self._k_ratio}}
        )
