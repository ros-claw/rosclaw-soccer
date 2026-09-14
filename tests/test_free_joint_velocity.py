"""Native oracle checks for the mixed-frame MuJoCo free-joint contract."""

import numpy as np
import pytest

from rosclaw_soccer.sim.free_joint_velocity import free_joint_body_velocity


@pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, np.pi, -1.2])
@pytest.mark.parametrize("axis", [[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]])
def test_matches_native_body_velocity(angle: float, axis: list[float]) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="root"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    direction = np.asarray(axis) / np.linalg.norm(axis)
    quaternion = np.r_[np.cos(angle / 2), direction * np.sin(angle / 2)]
    velocity = np.asarray([1.2, -0.7, 0.4, 0.3, -0.8, 0.6])
    data.qpos[3:7] = quaternion
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    native = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, native, 1)
    linear, angular = free_joint_body_velocity(
        world_from_body_quaternion_wxyz=quaternion, free_joint_qvel=velocity
    )
    np.testing.assert_allclose(linear, native[3:], atol=1e-12)
    np.testing.assert_allclose(angular, native[:3], atol=1e-12)
    np.testing.assert_array_equal(velocity, data.qvel)
    np.testing.assert_array_equal(quaternion, data.qpos[3:7])
    linear[:] = 0
    angular[:] = 0
    np.testing.assert_array_equal(velocity, data.qvel)


@pytest.mark.parametrize(
    "quaternion,velocity",
    [
        ([2, 0, 0, 0], [0] * 6),
        ([0, 0, 0, 0], [0] * 6),
        ([1, 0, 0], [0] * 6),
        ([1, 0, 0, 0], [0] * 5),
        ([np.nan, 0, 0, 0], [0] * 6),
        ([1, 0, 0, 0], [np.inf] * 6),
        ([True, False, False, False], [0] * 6),
        ([1, 0, 0, 0], [False] * 6),
        ([1j, 0, 0, 0], [0] * 6),
        ([1, 0, 0, 0], ["0"] * 6),
    ],
)
def test_rejects_invalid_state(quaternion: list, velocity: list) -> None:
    with pytest.raises(ValueError):
        free_joint_body_velocity(
            world_from_body_quaternion_wxyz=np.asarray(quaternion),
            free_joint_qvel=np.asarray(velocity),
        )


def test_quaternion_sign_and_readonly_input() -> None:
    quaternion = np.asarray([0.5, 0.5, 0.5, 0.5])
    velocity = np.arange(6, dtype=np.float64)
    quaternion.flags.writeable = False
    velocity.flags.writeable = False
    first = free_joint_body_velocity(
        world_from_body_quaternion_wxyz=quaternion, free_joint_qvel=velocity
    )
    second = free_joint_body_velocity(
        world_from_body_quaternion_wxyz=-quaternion, free_joint_qvel=velocity
    )
    for a, b in zip(first, second, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12)
