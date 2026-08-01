import numpy as np

from falcon.replay.rng import Rng


def test_same_seed_produces_same_named_streams():
    left = Rng(123)
    right = Rng(123)

    for name in ("global_init", "client_selection", "aggregation"):
        np.testing.assert_array_equal(
            left.stream(name).integers(0, 2**31, size=20),
            right.stream(name).integers(0, 2**31, size=20),
        )


def test_different_names_produce_different_streams():
    rng = Rng(123)

    assert not np.array_equal(
        rng.stream("client.a.optimizer").integers(0, 2**31, size=20),
        rng.stream("client.b.optimizer").integers(0, 2**31, size=20),
    )


def test_stream_creation_order_does_not_change_sequences():
    forward = Rng(456)
    reverse = Rng(456)
    first = forward.stream("client_selection").random(20)
    second = forward.stream("evaluation").random(20)

    second_reversed = reverse.stream("evaluation").random(20)
    first_reversed = reverse.stream("client_selection").random(20)

    np.testing.assert_array_equal(first, first_reversed)
    np.testing.assert_array_equal(second, second_reversed)


def test_state_dict_round_trips_mid_sequence():
    original = Rng(789)
    original.stream("aggregation").normal(size=13)
    original.stream("client.7.dataloader").integers(0, 10, size=7)
    snapshot = original.state_dict()

    expected_aggregation = original.stream("aggregation").normal(size=20)
    expected_dataloader = original.stream("client.7.dataloader").integers(
        0, 10, size=20
    )

    restored = Rng(0)
    restored.load_state_dict(snapshot)

    np.testing.assert_array_equal(
        expected_aggregation, restored.stream("aggregation").normal(size=20)
    )
    np.testing.assert_array_equal(
        expected_dataloader,
        restored.stream("client.7.dataloader").integers(0, 10, size=20),
    )
    np.testing.assert_array_equal(
        original.stream("evaluation").random(20),
        restored.stream("evaluation").random(20),
    )


def test_state_dict_returns_an_independent_snapshot():
    rng = Rng(321)
    rng.stream("aggregation").random(3)
    expected = rng.state_dict()

    snapshot = rng.state_dict()
    snapshot["root_seed"] = 999
    snapshot["streams"]["aggregation"]["state"]["state"] = 0

    assert rng.state_dict() == expected
