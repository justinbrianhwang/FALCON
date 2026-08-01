"""Unit tests for the runner overlay hook (Task T5).

The overlay is called at every stage boundary AFTER the stage computes and
AFTER failure injection, BEFORE recording and before downstream use. Default
``overlay=None`` must stay byte-identical to the pre-hook runner.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from falcon.pipeline.runner import run
from falcon.recorder.recorder import Recorder
from falcon.replay.rng import Rng
from falcon.schema import (
    AggregationConfig,
    CompressionConfig,
    DatasetConfig,
    LocalConfig,
    OutcomeState,
    RunConfig,
    SelectionConfig,
)

STAGE_ORDER = ("selection", "local", "compression", "aggregation", "evaluation")

#: committed golden boundary hashes for the ``cfg`` fixture below, generated
#: once at the T4/T5/T8-F fix HEAD (CONTRACTS v0.2 round-keyed streams)
GOLDEN_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_stage_hashes.json"


@pytest.fixture
def cfg() -> RunConfig:
    return RunConfig(
        run_id="test_overlay",
        seed=42,
        rounds=3,
        dataset=DatasetConfig(
            num_clients=5, num_features=10, num_classes=2, samples_per_client=80
        ),
        selection=SelectionConfig(clients_per_round=3),
        local=LocalConfig(lr=0.5, local_steps=8, batch_size=16),
        compression=CompressionConfig(kind="identity"),
        aggregation=AggregationConfig(rule="weighted_mean"),
    )


class _FixedOverlay:
    """Replaces the state at exactly one (round, stage) boundary."""

    def __init__(self, round_id: int, stage: str, state):
        self.round_id = round_id
        self.stage = stage
        self.state = state
        self.seen: list[tuple[int, str]] = []

    def override(self, round_id, stage, state):
        self.seen.append((round_id, stage))
        if (round_id, stage) == (self.round_id, self.stage):
            return self.state
        return state


def _recorded_baseline(cfg, root, run_id="baseline"):
    recorder = Recorder(root, run_id)
    outcomes = run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    return recorder, outcomes


def test_overlay_none_is_byte_identical(cfg, tmp_path):
    """Default None (and explicit None) reproduces the pre-hook runner exactly."""
    plain, outcomes_plain = _recorded_baseline(cfg, tmp_path, "plain")
    explicit_none, outcomes_none = _recorded_baseline(cfg, tmp_path, "none")
    assert plain.stage_hashes() == explicit_none.stage_hashes()
    assert [o.model_hash for o in outcomes_plain] == [
        o.model_hash for o in outcomes_none
    ]
    assert [o.metrics for o in outcomes_plain] == [o.metrics for o in outcomes_none]


def test_no_overlay_run_reproduces_golden_stage_hashes(cfg, tmp_path):
    """Finding-8 oracle: the runner is pinned to a committed golden fixture.

    ``test_overlay_none_is_byte_identical`` compares the runner against
    itself, so a common behavior change (an unconditional extra RNG draw, a
    renamed stream, a stage reorder) passes both sides. This test compares a
    fresh no-overlay run against boundary hashes generated once at the fix
    HEAD — any such drift breaks it.
    """
    golden = json.loads(GOLDEN_FIXTURE.read_text(encoding="utf-8"))
    recorder = Recorder(tmp_path, "golden")
    run(cfg, recorder=recorder, rng=Rng(cfg.seed))
    assert {
        f"{round_id}/{stage}": content_hash
        for (round_id, stage), content_hash in recorder.stage_hashes().items()
    } == golden["stage_hashes"]


def test_recording_overlay_sees_all_five_boundaries(cfg, tmp_path):
    """A pass-through overlay is called once per stage per round, in order."""
    calls: list[tuple[int, str]] = []

    class RecordingOverlay:
        def override(self, round_id, stage, state):
            calls.append((round_id, stage))
            return state

    tapped = Recorder(tmp_path, "tapped")
    run(cfg, recorder=tapped, rng=Rng(cfg.seed), overlay=RecordingOverlay())

    assert calls == [
        (round_id, stage)
        for round_id in range(cfg.rounds)
        for stage in STAGE_ORDER
    ]

    # pass-through changes nothing downstream or on disk
    plain, _ = _recorded_baseline(cfg, tmp_path, "plain")
    assert tapped.stage_hashes() == plain.stage_hashes()


def test_replace_selection_changes_which_clients_train(cfg, tmp_path):
    baseline, _ = _recorded_baseline(cfg, tmp_path, "baseline")
    original = baseline.load(1, "selection")
    bystanders = sorted(set(original.candidate_ids) - set(original.selected_ids))
    swapped = sorted(bystanders + [original.selected_ids[0]])
    assert swapped != original.selected_ids  # genuinely different subset

    replacement = original.model_copy(update={"selected_ids": swapped})
    overlay = _FixedOverlay(1, "selection", replacement)
    recorder = Recorder(tmp_path, "overlaid")
    run(cfg, recorder=recorder, rng=Rng(cfg.seed), overlay=overlay)

    assert recorder.load(1, "selection").selected_ids == swapped
    local_states = recorder.load(1, "local")
    assert [s.client_id for s in local_states] == swapped
    # other rounds are untouched
    assert recorder.load(0, "selection").selected_ids == baseline.load(0, "selection").selected_ids


def test_replace_local_changes_what_aggregation_sees(cfg, tmp_path):
    baseline, _ = _recorded_baseline(cfg, tmp_path, "baseline")
    zeroed = [
        state.model_copy(update={"update": np.zeros_like(state.update)})
        for state in baseline.load(1, "local")
    ]
    recorder = Recorder(tmp_path, "overlaid")
    outcomes = run(
        cfg,
        recorder=recorder,
        rng=Rng(cfg.seed),
        overlay=_FixedOverlay(1, "local", zeroed),
    )

    aggregation = recorder.load(1, "aggregation")
    assert np.array_equal(aggregation.aggregate, np.zeros_like(aggregation.aggregate))
    # zero update => model unchanged by round 1
    assert outcomes[1].model_hash == outcomes[0].model_hash
    # training resumes from the unchanged model in round 2
    assert outcomes[2].model_hash != outcomes[1].model_hash


def test_replace_compression_changes_what_aggregation_sees(cfg, tmp_path):
    baseline, _ = _recorded_baseline(cfg, tmp_path, "baseline")
    zeroed = [
        state.model_copy(update={"update": np.zeros_like(state.update)})
        for state in baseline.load(1, "compression")
    ]
    recorder = Recorder(tmp_path, "overlaid")
    outcomes = run(
        cfg,
        recorder=recorder,
        rng=Rng(cfg.seed),
        overlay=_FixedOverlay(1, "compression", zeroed),
    )

    aggregation = recorder.load(1, "aggregation")
    assert np.array_equal(aggregation.aggregate, np.zeros_like(aggregation.aggregate))
    assert outcomes[1].model_hash == outcomes[0].model_hash


def test_replace_aggregation_changes_the_model_update(cfg, tmp_path):
    baseline, _ = _recorded_baseline(cfg, tmp_path, "baseline")
    original = baseline.load(1, "aggregation")
    replacement = original.model_copy(
        update={"aggregate": np.zeros_like(original.aggregate)}
    )
    recorder = Recorder(tmp_path, "overlaid")
    outcomes = run(
        cfg,
        recorder=recorder,
        rng=Rng(cfg.seed),
        overlay=_FixedOverlay(1, "aggregation", replacement),
    )

    recorded = recorder.load(1, "aggregation")
    assert np.array_equal(recorded.aggregate, replacement.aggregate)
    assert outcomes[1].model_hash == outcomes[0].model_hash


def test_replace_evaluation_only_changes_the_recorded_outcome(cfg, tmp_path):
    _, baseline_outcomes = _recorded_baseline(cfg, tmp_path, "baseline")
    crafted = OutcomeState(
        round_id=1,
        model_hash="0" * 64,
        metrics={"accuracy": 0.25, "loss": 7.5},
        per_class={},
    )
    recorder = Recorder(tmp_path, "overlaid")
    outcomes = run(
        cfg,
        recorder=recorder,
        rng=Rng(cfg.seed),
        overlay=_FixedOverlay(1, "evaluation", crafted),
    )

    # the replacement is what gets recorded and returned for round 1
    recorded = recorder.load(1, "evaluation")
    assert recorded.metrics == crafted.metrics
    assert recorded.model_hash == crafted.model_hash
    assert outcomes[1].metrics == crafted.metrics
    # training is unaffected: every other model hash matches the plain baseline
    assert [o.model_hash for o in outcomes if o.round_id != 1] == [
        o.model_hash for o in baseline_outcomes if o.round_id != 1
    ]


def test_selection_overlay_cannot_contaminate_later_round_rng(cfg, tmp_path):
    """Finding-1 regression (round-2 review): participation-history coupling.

    The overlay replaces the round-0 selection with the round-1 subset AND
    restores the round-0 aggregate exactly, so the model entering round 1 is
    bit-identical to the baseline's. With round-keyed dataloader streams
    (CONTRACTS §3 v0.2, ``client.<id>.round.<t>.dataloader``) every round-1
    client update must then be identical too. Under the old persistent
    per-client streams, the client selected one extra time in round 0 had its
    stream advanced and produced a DIFFERENT round-1 update — a selection
    intervention measured RNG drift, not state causation.
    """
    baseline, baseline_outcomes = _recorded_baseline(cfg, tmp_path, "baseline")
    round0_selection = baseline.load(0, "selection").selected_ids
    round1_selection = baseline.load(1, "selection").selected_ids
    assert round1_selection != round0_selection  # the scenario needs a real swap
    baseline_aggregate = baseline.load(0, "aggregation")

    class _SelectionRestoreOverlay:
        def override(self, round_id, stage, state):
            if (round_id, stage) == (0, "selection"):
                return state.model_copy(update={"selected_ids": list(round1_selection)})
            if (round_id, stage) == (0, "aggregation"):
                return baseline_aggregate  # restore the model exactly
            return state

    recorder = Recorder(tmp_path, "overlaid")
    outcomes = run(
        cfg, recorder=recorder, rng=Rng(cfg.seed), overlay=_SelectionRestoreOverlay()
    )

    # the aggregate restore really did repair the model entering round 1
    assert outcomes[0].model_hash == baseline_outcomes[0].model_hash
    # later selection is untouched
    assert recorder.load(1, "selection").selected_ids == round1_selection
    # every client's round-1 update is bit-identical to the baseline's — even
    # for the clients the overlay selected one extra time in round 0
    baseline_local = {s.client_id: s for s in baseline.load(1, "local")}
    assert set(baseline_local) == set(round1_selection)
    for state in recorder.load(1, "local"):
        np.testing.assert_array_equal(
            state.update, baseline_local[state.client_id].update
        )
