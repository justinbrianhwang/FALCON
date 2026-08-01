import json
from enum import Enum

import numpy as np
import pytest
from pydantic import BaseModel

from falcon.recorder.hashing import hash_array, hash_model
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


def test_metadata_rejects_non_json_native_config_values(tmp_path):
    metadata = RunMetadata(
        run_id="metadata", seed=12, rounds=4, config={"shape": (1, 2)}
    )

    with pytest.raises(ValueError, match="field 'config'.*tuple"):
        Recorder(tmp_path, "metadata").save_metadata(metadata)


def test_non_finite_metrics_round_trip_with_stable_hash(tmp_path):
    recorder = Recorder(tmp_path, "non-finite")
    state = OutcomeState(
        round_id=0,
        model_hash="model",
        metrics={
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
        },
    )

    recorder.record(0, "evaluation", state)
    loaded = recorder.load(0, "evaluation")

    assert isinstance(loaded, OutcomeState)
    assert np.isnan(loaded.metrics["nan"])
    assert loaded.metrics["positive"] == float("inf")
    assert loaded.metrics["negative"] == float("-inf")
    assert loaded.content_hash == hash_model(state)
    data = json.loads(
        (tmp_path / "runs" / "non-finite" / "round_0" / "evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["metrics"] == {
        "nan": {"__falcon_float__": "NaN"},
        "negative": {"__falcon_float__": "-Infinity"},
        "positive": {"__falcon_float__": "Infinity"},
    }


def test_hash_array_includes_dtype_shape_and_values():
    assert hash_array(np.zeros(1, np.float64)) != hash_array(
        np.zeros(8, np.uint8)
    )
    flat = np.arange(4, dtype=np.int64)
    assert hash_array(flat) != hash_array(flat.reshape(2, 2))
    changed = flat.astype(np.float64)
    changed[0] = np.nextafter(changed[0], np.inf)
    assert hash_array(flat.astype(np.float64)) != hash_array(changed)


def test_hash_model_is_sensitive_to_array_scalar_and_dict_changes():
    states = _states()

    array_changed = states["local"][0].model_copy(deep=True)
    array_changed.update[0] = np.nextafter(array_changed.update[0], np.inf)
    assert hash_model(states["local"][0]) != hash_model(array_changed)

    scalar_changed = states["local"][0].model_copy(deep=True)
    scalar_changed.num_examples += 1
    assert hash_model(states["local"][0]) != hash_model(scalar_changed)

    dict_changed = states["selection"].model_copy(deep=True)
    dict_changed.rng_state["position"] += 1
    assert hash_model(states["selection"]) != hash_model(dict_changed)


class _CompressionKind(Enum):
    IDENTITY = "identity"


@pytest.mark.parametrize(
    "value, detail",
    [
        ((1, 2), "tuple"),
        (_CompressionKind.IDENTITY, "Enum"),
        ({1: "value"}, "non-string dictionary key"),
    ],
)
def test_record_rejects_non_json_native_any_values(tmp_path, value, detail):
    state = _states()["compression"][0].model_copy(deep=True)
    state.compression_params = {"invalid": value}

    with pytest.raises(
        ValueError, match=rf"field 'compression_params'.*{detail}"
    ):
        Recorder(tmp_path, "invalid-any").record(0, "compression", [state])


def test_record_rejects_reserved_array_marker_in_user_data(tmp_path):
    state = _states()["compression"][0].model_copy(deep=True)
    state.compression_params = {"__falcon_array__": "user-value"}

    with pytest.raises(ValueError, match="compression_params.*reserved key"):
        Recorder(tmp_path, "reserved-marker").record(
            0, "compression", [state]
        )


def test_restore_only_treats_complete_array_references_as_arrays():
    value = {"__falcon_array__": "user-value", "other": 1}

    assert Recorder._restore_value(value, {}) == value


@pytest.mark.parametrize("round_id", [True, False, 1.5, "1", None])
def test_record_and_load_require_integer_round_ids(tmp_path, round_id):
    recorder = Recorder(tmp_path, "round-id")

    with pytest.raises(TypeError, match="round_id must be an int"):
        recorder.record(round_id, "selection", _states()["selection"])
    with pytest.raises(TypeError, match="round_id must be an int"):
        recorder.load(round_id, "selection")


@pytest.mark.parametrize("stage", ["local", "compression"])
def test_per_client_stages_require_state_lists(tmp_path, stage):
    state = _states()[stage][0]

    with pytest.raises(TypeError, match="requires a state list"):
        Recorder(tmp_path, f"bare-{stage}").record(0, stage, state)


def test_scalar_stage_rejects_state_list(tmp_path):
    with pytest.raises(TypeError, match="does not accept a state list"):
        Recorder(tmp_path, "scalar-list").record(
            0, "selection", [_states()["selection"]]
        )


def test_unknown_stage_is_rejected(tmp_path):
    recorder = Recorder(tmp_path, "unknown-stage")

    with pytest.raises(ValueError, match="unknown stage"):
        recorder.record(0, "unknown", _states()["selection"])
    with pytest.raises(ValueError, match="unknown stage"):
        recorder.load(0, "unknown")


def test_traversal_client_id_is_rejected(tmp_path):
    state = _states()["local"][0].model_copy(update={"client_id": "../c0"})

    with pytest.raises(ValueError, match="invalid client_id"):
        Recorder(tmp_path, "traversal").record(0, "local", [state])


def test_casefold_duplicate_client_ids_are_rejected(tmp_path):
    first = _states()["local"][0].model_copy(update={"client_id": "client"})
    second = _states()["local"][1].model_copy(update={"client_id": "CLIENT"})

    with pytest.raises(ValueError, match="duplicate client_id"):
        Recorder(tmp_path, "duplicate").record(0, "local", [first, second])


@pytest.mark.parametrize("client_id", ["client.", "client "])
def test_trailing_dot_or_space_client_id_is_rejected(tmp_path, client_id):
    state = _states()["local"][0].model_copy(update={"client_id": client_id})

    with pytest.raises(ValueError, match="invalid client_id"):
        Recorder(tmp_path, "trailing").record(0, "local", [state])


def test_tampered_json_fails_content_hash_validation(tmp_path):
    recorder = Recorder(tmp_path, "tampered")
    recorder.record(3, "evaluation", _states()["evaluation"])
    path = tmp_path / "runs" / "tampered" / "round_3" / "evaluation.json"
    original = path.read_text(encoding="utf-8")
    tampered = original.replace('"loss": 0.5', '"loss": 0.6')
    assert tampered != original
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        recorder.load(3, "evaluation")
