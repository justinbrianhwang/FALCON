"""L1 — local-training failure: a subset of clients trains with a wrong learning rate."""
from __future__ import annotations

import math

from ..base import FailureInjector


class LrMisconfigInjector(FailureInjector):
    """Multiply the local learning rate of affected clients while active.

    Parameters (``spec.parameters``):
      - exactly one of ``affected_clients`` (explicit id list) or ``fraction``
        (share of the partition's clients, chosen once at construction from
        stream ``failure.local``: ``max(1, round(fraction * K))`` ids drawn
        without replacement, sorted);
      - ``lr_multiplier``: applied as ``lr * multiplier`` (e.g. 10.0, 0.01;
        negative allowed as a sanity case). Must be finite — a NaN/inf
        multiplier would silently poison every downstream stage state.
    """

    stage = "local"

    def __init__(self, spec, partition, rng):
        super().__init__(spec, partition, rng)
        params = spec.parameters
        self._multiplier = float(params["lr_multiplier"])
        if not math.isfinite(self._multiplier):
            raise ValueError(f"lr_multiplier must be finite, got {self._multiplier}")
        explicit = params.get("affected_clients")
        fraction = params.get("fraction")
        if (explicit is None) == (fraction is None):
            raise ValueError(
                "lr_misconfig needs exactly one of 'affected_clients' or 'fraction'"
            )
        if explicit is not None:
            unknown = sorted(set(explicit) - set(partition))
            if unknown:
                raise ValueError(f"affected_clients not in partition: {unknown}")
            self._affected = frozenset(str(cid) for cid in explicit)
        else:
            fraction = float(fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")
            ids = sorted(partition)
            n_affected = max(1, min(len(ids), int(round(fraction * len(ids)))))
            chosen = self._gen().choice(len(ids), size=n_affected, replace=False)
            self._affected = frozenset(ids[int(i)] for i in chosen)

    @property
    def affected_clients(self) -> frozenset[str]:
        return self._affected

    def local_cfg(self, client_id, cfg, round_id):
        if not self.active(round_id) or client_id not in self._affected:
            return cfg.model_copy()
        return cfg.model_copy(update={"lr": cfg.lr * self._multiplier})
