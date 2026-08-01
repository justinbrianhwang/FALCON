import json

import numpy as np
from pydantic import BaseModel

from falcon.recorder.hashing import hash_model
from falcon.recorder.recorder import Recorder
from falcon.schema import (
    AggregationState,
    ClientLocalState,
    CompressionState,
    OutcomeState,
    RunMetadata,
    SelectionState,
)


def _states():
    return {
        "selection": SelectionState(
            round_id=3,
            candidate_ids=["c0", "c1", "c2"],
            selected_ids=["c2", "c0"],
            sampling_probs={"c0": 1 / 3, "c1": 1 / 3, "c2": 1 / 3},
            rng_state={"position": 17},
        ),
        "local": [
            ClientLocalState(
                round_id=3,
                client_id=client_id,
                base_model_hash="base",
                update=np.array([index + 0.25, index - 0.5], dtype=np.float64),
                num_examples=10 + index,
                num_steps=2,
                loss_history=[0.9, 0.7],
                rng_state={"position": index},
            )
            for index, client_id in enumerate(("client.2", "c0"))
        ],
        "compression": [
            CompressionState(
                round_id=3,
                client_id=client_id,
                uncompressed_hash="update",
                update=np.array([index, -index], dtype=np.float64),
                compression_params={"kind": "identity"},
                bytes_transmitted=16,
            )
            for index, client_id in enumerate(("client.2", "c0"), start=1)
        ],
        "aggregation": AggregationState(
            round_id=3,
            received_ids=["c2", "c0"],
            accepted_ids=["c2", "c0"],
            rejected_ids=[],
            weights={"c2": 0.6, "c0": 0.4},
            aggregate=np.array([1.5, -2.25], dtype=np.float64),
        ),
        "evaluation": OutcomeState(
            round_id=3,
            model_hash="model",
            metrics={"accuracy": 0.75, "loss": 0.5},
            per_class={"0": {"accuracy": 0.8}},
        ),
    }


def _assert_model_equal(expected: BaseModel, actual: BaseModel):
    assert type(actual) is type(expected)
    for name in type(expected).model_fields:
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if isinstance(expected_value, np.ndarray):
            assert actual_value.dtype == expected_value.dtype
            assert actual_value.shape == expected_value.shape
            assert actual_value.tobytes() == expected_value.tobytes()
        elif name != "content_hash":
            assert actual_value == expected_value
    assert actual.content_hash == hash_model(expected)


def test_record_load_round_trip_for_every_stage_state(tmp_path):
    recorder = Recorder(tmp_path, "round-trip")

    for stage, state in _states().items():
        recorder.record(3, stage, state)
        loaded = recorder.load(3, stage)
        if isinstance(state, list):
            assert isinstance(loaded, list)
            assert [item.client_id for item in loaded] == [
                item.client_id for item in state
            ]
            for expected, actual in zip(state, loaded):
                _assert_model_equal(expected, actual)
        else:
            assert isinstance(loaded, BaseModel)
            _assert_model_equal(state, loaded)


def test_content_hash_is_stable_across_saves(tmp_path):
    first = Recorder(tmp_path, "first")
    second = Recorder(tmp_path, "second")
    first_states = _states()
    second_states = _states()

    for stage in first_states:
        first.record(3, stage, first_states[stage])
        second.record(3, stage, second_states[stage])

    assert first.stage_hashes() == second.stage_hashes()


def test_metadata_is_saved_as_json(tmp_path):
    recorder = Recorder(tmp_path, "metadata")
    metadata = RunMetadata(run_id="metadata", seed=12, rounds=4, config={"x": 1})

    recorder.save_metadata(metadata)

    path = tmp_path / "runs" / "metadata" / "metadata.json"
    assert json.loads(path.read_text(encoding="utf-8")) == metadata.model_dump(
        mode="json"
    )
