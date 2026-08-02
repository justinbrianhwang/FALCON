"""Shared deterministic client-targeting rules for failure injectors (Task T17)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from falcon.pipeline.synthetic_data import ClientData


def minority_heavy_clients(
    partition: dict[str, "ClientData"], target_class: int
) -> frozenset[str]:
    """Clients whose within-client share of ``target_class`` samples exceeds the
    dataset-wide share of that class (i.e. the uniform share the class would
    hold if its samples were spread evenly across the whole partition).

    The same "minority-heavy" rule backs selection's ``minority_exclusion``
    and aggregation's ``wrong_sample_weights`` ``"biased"`` mode. Computed
    once, deterministically, from ``partition``; no RNG involved.
    """
    counts = {
        cid: int((data.y == target_class).sum())
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
