import numpy as np
import pytest

from rosclaw_soccer.training.receiving_oracle_continuation import (
    locked_knot_count,
    propose_continuations,
)
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)


def schedule():
    return ReceivingOracleSchedule(
        "blue.playmaker", "A0_leg12", 5, 10, tuple((0.2,) * 12 for _ in range(12))
    )


@pytest.mark.parametrize(
    "branch,locked", [(0, 0), (5, 0), (6, 1), (7, 2), (15, 2), (16, 2), (17, 3)]
)
def test_interpolation_dependency(branch, locked):
    assert locked_knot_count(schedule(), branch_frame=branch) == locked


@pytest.mark.parametrize("branch", [0, 5, 6, 7, 15, 16, 17, 50, 83])
def test_actual_filtered_prefix_is_unchanged(branch):
    original = schedule()
    proposals = propose_continuations(original, branch_frame=branch, seed=7)
    assert len(proposals) == 5 and proposals[0] is original
    assert proposals == propose_continuations(original, branch_frame=branch, seed=7)
    histories = []
    for proposal in proposals:
        cursor = ReceivingOracleCursor(proposal)
        history = []
        for frame in range(120):
            result = cursor.step(frame, active=frame % 9 != 0, predecessor=np.full(12, 0.03))
            history.append(np.zeros(12) if result is None else result)
        histories.append(np.asarray(history))
    for history in histories[1:]:
        np.testing.assert_array_equal(history[:branch], histories[0][:branch])
        assert np.max(np.abs(history)) <= 0.1
        assert not np.array_equal(history[branch:], histories[0][branch:])


@pytest.mark.parametrize(
    "kwargs", [{"seed": True}, {"seed": -1}, {"count": 0}, {"standard_deviation": np.nan}]
)
def test_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        propose_continuations(schedule(), branch_frame=15, **{"seed": 1, **kwargs})


def test_cannot_rewrite_past_final_knot():
    with pytest.raises(ValueError, match="no unexecuted"):
        propose_continuations(schedule(), branch_frame=125, seed=1)


def test_reject_boolean_time():
    with pytest.raises(ValueError):
        locked_knot_count(schedule(), branch_frame=True)
