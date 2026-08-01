"""Passive stage-anomaly localization baseline (Plan §19.2, Task T8).

Scores each intervenable stage from RECORDED states only — no replay, no
interventions — and localizes the failure to the argmax stage. These are the
rivals FALCON must beat in E1/E2, so scoring is implemented fairly: bounded
deviations, exact client matching, deterministic tie-breaks.
"""
from pathlib import Path

import numpy as np

from falcon.recorder import Recorder
from falcon.schema import STAGES

#: Intervenable stages in canonical pipeline order. ``evaluation`` is
#: observed, not intervened on, so it is never scored. This order is also
#: the deterministic tie-break order used by :func:`passive_localize`.
INTERVENABLE_STAGES: tuple[str, ...] = tuple(s for s in STAGES if s != "evaluation")


def _round_ids(root: Path, run_id: str) -> list[int]:
    """Sorted round ids recorded under ``root/runs/<run_id>``.

    Derived from the on-disk ``round_*`` directories so it works whether or
    not the producer saved ``metadata.json``.
    """
    run_dir = Path(root) / "runs" / run_id
    ids: list[int] = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            try:
                ids.append(int(child.name[len("round_"):]))
            except ValueError:
                continue
    return sorted(ids)


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Bounded relative L2 deviation ``||a-b|| / (||a|| + ||b||)`` in [0, 1].

    0.0 iff the vectors are equal; 1.0 is the maximum (e.g. one side is all
    zeros while the other is not). A bounded deviation keeps the metric
    consistent with the unmatched-client rule, which also contributes the
    maximum deviation 1.0. 0/0 (both vectors zero) is defined as 0.0.
    """
    denom = float(np.linalg.norm(a) + np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def _jaccard_distance(a: list[str], b: list[str]) -> float:
    """1 - |A ∩ B| / |A ∪ B| over id sets; two empty sets count as identical."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)


def _per_client_deviations(ref_states: list, fail_states: list) -> list[float]:
    """Relative L2 deviation per client, matched by client id.

    Clients present in only one run contribute the maximum deviation 1.0.
    """
    ref_by_id = {s.client_id: s.update for s in ref_states}
    fail_by_id = {s.client_id: s.update for s in fail_states}
    deviations: list[float] = []
    for client_id in sorted(set(ref_by_id) | set(fail_by_id)):
        if client_id in ref_by_id and client_id in fail_by_id:
            deviations.append(_relative_l2(ref_by_id[client_id], fail_by_id[client_id]))
        else:
            deviations.append(1.0)
    return deviations


def passive_stage_scores(
    ref_root: Path,
    fail_root: Path,
    reference_run_id: str,
    failure_run_id: str,
) -> dict[str, float]:
    """Stage-anomaly scores for a reference/failure run pair, from recorded states.

    Both roots follow the recorder layout ``<root>/runs/<run_id>/``. Only
    rounds recorded in BOTH runs are compared. Scores per intervenable stage:

    - selection: Jaccard distance between ``selected_ids`` per round, averaged;
    - local: mean over rounds/clients of the relative L2 deviation of updates
      (matched by client id; unmatched clients count as max deviation 1.0);
    - compression: same on post-compression updates;
    - aggregation: relative L2 deviation of the aggregate vector per round,
      averaged.

    Returns a dict mapping every stage in :data:`INTERVENABLE_STAGES` to a
    score in [0, 1]. A stage with no comparable data scores 0.0.
    """
    ref = Recorder(Path(ref_root), reference_run_id)
    fail = Recorder(Path(fail_root), failure_run_id)
    rounds = sorted(
        set(_round_ids(ref_root, reference_run_id))
        & set(_round_ids(fail_root, failure_run_id))
    )

    selection_distances: list[float] = []
    local_deviations: list[float] = []
    compression_deviations: list[float] = []
    aggregation_deviations: list[float] = []
    for round_id in rounds:
        sel_ref = ref.load(round_id, "selection")
        sel_fail = fail.load(round_id, "selection")
        selection_distances.append(
            _jaccard_distance(sel_ref.selected_ids, sel_fail.selected_ids)
        )
        local_deviations.extend(
            _per_client_deviations(ref.load(round_id, "local"), fail.load(round_id, "local"))
        )
        compression_deviations.extend(
            _per_client_deviations(
                ref.load(round_id, "compression"), fail.load(round_id, "compression")
            )
        )
        agg_ref = ref.load(round_id, "aggregation")
        agg_fail = fail.load(round_id, "aggregation")
        aggregation_deviations.append(_relative_l2(agg_ref.aggregate, agg_fail.aggregate))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "selection": _mean(selection_distances),
        "local": _mean(local_deviations),
        "compression": _mean(compression_deviations),
        "aggregation": _mean(aggregation_deviations),
    }


def passive_localize(scores: dict[str, float]) -> str:
    """Argmax stage of ``passive_stage_scores`` output.

    Deterministic tie-break: the earliest stage in STAGES order wins.
    """
    candidates = [s for s in INTERVENABLE_STAGES if s in scores]
    if not candidates:
        raise ValueError(f"no intervenable stage present in scores: {sorted(scores)}")
    # max() keeps the first maximal element, and candidates is in STAGES order.
    return max(candidates, key=lambda s: scores[s])
