"""Minimal T1 replay check; the full pipeline replay coverage lands in T2."""

import numpy as np

from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import AggregationState, SelectionState


def _record_synthetic_states(root, run_id, seed=2025):
    rng = Rng(seed)
    recorder = Recorder(root, run_id)
    selected = rng.stream("client_selection").choice(5, size=2, replace=False)
    recorder.record(
        0,
        "selection",
        SelectionState(
            round_id=0,
            candidate_ids=[f"c{i}" for i in range(5)],
            selected_ids=[f"c{i}" for i in selected],
            sampling_probs={f"c{i}": 0.2 for i in range(5)},
            rng_state=rng.state_dict(),
        ),
    )
    recorder.record(
        0,
        "aggregation",
        AggregationState(
            round_id=0,
            received_ids=[f"c{i}" for i in selected],
            accepted_ids=[f"c{i}" for i in selected],
            rejected_ids=[],
            weights={f"c{i}": 0.5 for i in selected},
            aggregate=rng.stream("aggregation").normal(size=4).astype(np.float64),
        ),
    )
    return recorder.stage_hashes()


# placeholder — superseded by full pipeline replay test (T2-F)
def test_same_seed_produces_equal_recorded_stage_hashes(tmp_path):
    assert _record_synthetic_states(tmp_path, "clean-a") == _record_synthetic_states(
        tmp_path, "clean-b"
    )


def test_different_seed_produces_different_recorded_stage_hashes(tmp_path):
    assert _record_synthetic_states(
        tmp_path, "seed-a", seed=2025
    ) != _record_synthetic_states(tmp_path, "seed-b", seed=2026)
