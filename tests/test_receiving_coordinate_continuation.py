import numpy as np
import pytest

from rosclaw_soccer.training.receiving_oracle_continuation import coordinate_continuations
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)


def schedule():
    return ReceivingOracleSchedule("blue.defender", "A3_sonic_residual", 0, 20, ((0.0,) * 29,) * 16)


@pytest.mark.parametrize("branch", [0, 1, 20, 40, 60, 81])
def test_coordinate_changes_only_future_named_joint(branch):
    source = schedule()
    proposals = coordinate_continuations(source, branch_frame=branch, dimensions=(4, 10))
    assert len(proposals) == 5 and proposals[0] is source
    for candidate, dimension in zip(proposals[1:], (4, 4, 10, 10), strict=True):
        old, new = ReceivingOracleCursor(source), ReceivingOracleCursor(candidate)
        for frame in range(160):
            expected = old.step(frame, active=True, predecessor=np.zeros(12))
            actual = new.step(frame, active=True, predecessor=np.zeros(12))
            assert expected is not None and actual is not None
            if frame < branch:
                np.testing.assert_array_equal(actual, expected)
            assert np.max(np.abs(actual)) <= 0.1
            np.testing.assert_array_equal(
                np.delete(actual, dimension), np.delete(expected, dimension)
            )


@pytest.mark.parametrize("dimensions", [(), (True,), (-1,), (29,), (4, 4), [4]])
def test_reject_invalid_coordinates(dimensions):
    with pytest.raises(ValueError):
        coordinate_continuations(schedule(), branch_frame=40, dimensions=dimensions)


@pytest.mark.parametrize("step", [True, 0, -0.1, 0.51, float("nan")])
def test_reject_invalid_step(step):
    with pytest.raises(ValueError):
        coordinate_continuations(schedule(), branch_frame=40, dimensions=(4,), normalized_step=step)
