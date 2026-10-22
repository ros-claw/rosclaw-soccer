import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_transfer_curriculum import initial_ball_offset, train_ball_states


def test_complete40_train_initial_states_not160_independent_courses():
    directions = {i: np.array([0.0, -1.0]) for i in (1, 4, 20, 21)}
    rows = train_ball_states(directions)
    assert len(rows) == 40 and [r["state_id"] for r in rows] == list(range(40))
    for context in directions:
        part = [r for r in rows if r["context"] == context]
        assert len(part) == 10
        assert len({tuple(r["ball_planar_offset_world"]) for r in part}) == 10
        assert sum(r["side"] < 0 for r in part) == 5
        assert all(r["bank"] == "TRAIN" and r["not_fresh"] for r in part)
        assert all(r["ball_planar_offset_world"][1] == 0 for r in part)
    assert all(np.linalg.norm(r["ball_planar_offset_world"]) > 0 for r in rows)


def test_initial_offset_is_owned_and_no_default_change():
    assert initial_ball_offset(None) is None
    value = np.array([0.03, 0.04])
    result = initial_ball_offset(value)
    value[:] = 1
    np.testing.assert_array_equal(result, [0.03, 0.04])


@pytest.mark.parametrize(
    "value",
    [
        [0.01, 0],
        np.zeros(3),
        np.array([0.0501, 0.0]),
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([0, 0]),
    ],
)
def test_no_clipping_or_invalid_initialization(value):
    with pytest.raises(ValueError):
        initial_ball_offset(value)


def test_direction_bank_must_be_complete_and_unit():
    with pytest.raises(ValueError):
        train_ball_states({1: np.array([1.0, 0.0])})
    with pytest.raises(ValueError):
        train_ball_states({i: np.array([2.0, 0.0]) for i in (1, 4, 20, 21)})
