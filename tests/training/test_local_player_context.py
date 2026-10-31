import numpy as np
import pytest

from rosclaw_soccer.training.local_player_context import nearest_player_context


def test_relative_state_in_local_frame():
    context = nearest_player_context(
        [1, 1], [1, 0], np.pi / 2, ["near", "far"], [[1, 2], [8, 8]], [[1, 2], [0, 0]]
    )
    assert np.allclose(context, [1, 1, 0, 2, 0], atol=1e-12)


def test_ties_are_roster_order_independent():
    a = nearest_player_context([0, 0], [0, 0], 0, ["b", "a"], [[1, 0], [-1, 0]], [[0, 0], [0, 0]])
    b = nearest_player_context([0, 0], [0, 0], 0, ["a", "b"], [[-1, 0], [1, 0]], [[0, 0], [0, 0]])
    assert np.array_equal(a, b)
    assert np.array_equal(a, [1, -1, 0, 0, 0])


def test_empty_and_invalid_inputs():
    assert np.array_equal(
        nearest_player_context([0, 0], [0, 0], 0, [], np.zeros((0, 2)), np.zeros((0, 2))),
        np.zeros(5),
    )
    with pytest.raises(ValueError):
        nearest_player_context([0, 0], [0, 0], 0, ["a", "a"], [[1, 0], [2, 0]], [[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        nearest_player_context([0, 0], [0, 0], 0, ["a"], [[float("nan"), 0]], [[0, 0]])
