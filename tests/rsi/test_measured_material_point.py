import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.measured_material_point import measure_material_point


def test_current_material_point_all_hg_columns_match_fixed_point_finite_difference():
    w = world()
    layout = w.layouts["a/"]
    w.data.qpos[layout.qi] = np.linspace(-0.2, 0.2, 29)
    # Deliberately leave the donor's FK cache stale.
    before = w.data.qpos.copy(), w.data.xpos.copy(), w.data.qvel.copy()
    local = np.array([0.13, -0.07, 0.09], np.float64)
    measured = measure_material_point(
        w.model, w.data, layout, body_name="a/link28", local_point=local
    )
    numerical = np.empty((3, 29))
    for j, qi in enumerate(layout.qi):
        points = []
        for sign in (-1, 1):
            scratch = mujoco.MjData(w.model)
            scratch.qpos[:] = w.data.qpos
            scratch.qpos[qi] += sign * 1e-6
            mujoco.mj_kinematics(w.model, scratch)
            body = w.model.body("a/link28").id
            points.append(scratch.xpos[body] + scratch.xmat[body].reshape(3, 3) @ local)
        numerical[:, j] = (points[1] - points[0]) / 2e-6
    np.testing.assert_allclose(measured["joint_jacobian_hg"], numerical, atol=1e-9, rtol=0)
    assert np.linalg.norm(numerical) > 0.1
    assert measured["joint_jacobian_hg"].shape == (3, 29)
    assert not set(layout.vi) & set(range(layout.rv, layout.rv + 6))
    assert not set(layout.vi) & set(range(layout.bv, layout.bv + 6))
    for a, b in zip(before, (w.data.qpos, w.data.xpos, w.data.qvel), strict=True):
        np.testing.assert_array_equal(a, b)
    assert w.data.time == 0 and measured["new_physical_steps"] == 0


@pytest.mark.parametrize("body", ["ball", "b/link28", "world"])
def test_other_player_ball_and_world_are_not_joint_authority(body):
    w = world()
    with pytest.raises(ValueError, match="selected player's"):
        measure_material_point(
            w.model, w.data, w.layouts["a/"], body_name=body, local_point=np.zeros(3)
        )


def test_invalid_local_point_is_rejected():
    w = world()
    with pytest.raises(ValueError):
        measure_material_point(
            w.model, w.data, w.layouts["a/"], body_name="a/link28", local_point=np.full(3, np.nan)
        )
