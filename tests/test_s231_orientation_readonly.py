import numpy as np
import pytest

from rosclaw_soccer.skills.team.independent_team_world import _pelvis_yaw, _roll_pitch


@pytest.mark.parametrize("helper", [_pelvis_yaw, _roll_pitch])
def test_orientation_queries_do_not_write_through_physics_views(helper):
    state = np.array((1.0, 2.0, 3.0, 1.8, 0.2, 0.1, 0.3))
    original = state.copy()
    result = helper(state[3:])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(state, original)
    state.setflags(write=False)
    np.testing.assert_allclose(helper(state[3:]), result)
