import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.ball_surface_velocity import inspect_ball_surface_velocity


def world():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="ball" pos="0 0 0.11"><freejoint/><geom name="ball_geom" type="sphere" size="0.11" mass="0.43"/></body></worldbody></mujoco>'
    )


def test_pure_rolling_has_zero_bottom_velocity_without_advancing_time():
    model = world()
    data = mujoco.MjData(model)
    data.qvel[:] = [1, 0, 0, 0, 1 / 0.11, 0]
    q, v = data.qpos.copy(), data.qvel.copy()
    result = inspect_ball_surface_velocity(model, data, geom_id=0)
    np.testing.assert_allclose(
        result["hypothetical_bottom_point_velocity_world_m_s"], [0, 0, 0], atol=1e-14
    )
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, v)
    assert data.time == 0


def test_rotated_body_angular_velocity_is_transformed_to_world():
    model = world()
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    # Local +x angular velocity rotates to world +y, rolling along world +x.
    data.qvel[:] = [1, 0, 0, 1 / 0.11, 0, 0]
    result = inspect_ball_surface_velocity(model, data, geom_id=0)
    np.testing.assert_allclose(result["angular_velocity_world_rad_s"], [0, 1 / 0.11, 0], atol=1e-14)
    np.testing.assert_allclose(
        result["hypothetical_bottom_point_velocity_world_m_s"], [0, 0, 0], atol=1e-14
    )


def test_sliding_and_airborne_state_are_not_misreported_as_rolling_contact():
    model = world()
    data = mujoco.MjData(model)
    data.qpos[2] = 2
    data.qvel[:3] = [1, 2, 0]
    result = inspect_ball_surface_velocity(model, data, geom_id=0)
    assert result["hypothetical_bottom_horizontal_speed_m_s"] == pytest.approx(np.sqrt(5))
    assert not result["ground_contact_established_by_this_function"]
