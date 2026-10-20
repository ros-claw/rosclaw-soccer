"""Private FK/Jacobian math only, not a trained-policy physics experiment."""

import math

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_velocity_observation import (
    G1VelocityObservation,
    world_command_to_body,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes


def fixture():
    children = "".join(
        f'<body name="j{i}" pos="0 0 0.01"><joint name="{name}" axis="0 1 0"/>'
        '<geom type="sphere" size="0.01" mass=".1"/></body>'
        for i, name in enumerate(G1_DDS_JOINT_NAMES)
    )
    motors = "".join(f'<motor joint="{name}"/>' for name in G1_DDS_JOINT_NAMES)
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        + children
        + '</body><body name="ball"><freejoint/>'
        '<geom type="sphere" size=".11" mass=".43"/></body></worldbody><actuator>'
        + motors
        + "</actuator></mujoco>"
    )
    data = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, data)
    return m, hash_bytes(data.tobytes())


def test_angular_observation_is_current_body_frame_not_world_or_cached_solver_data():
    m, h = fixture()
    sensor = G1VelocityObservation(m, expected_world_hash=h)
    q, v = np.zeros(43), np.zeros(41)
    q[2] = 0.793
    q[39] = 1.0
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    q[3:7] = [math.cos(0.5), *(axis * math.sin(0.5))]
    v[3:6] = [0.3, -0.2, 0.1]
    v[6:35] = 1.0
    originalq, originalv = q.copy(), v.copy()
    result = sensor.angular_velocity_body(q, v)
    np.testing.assert_allclose(result, v[3:6], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(q, originalq)
    np.testing.assert_array_equal(v, originalv)
    with pytest.raises(ValueError):
        G1VelocityObservation(m, expected_world_hash="sha256:" + "0" * 64)


def test_world_to_body_command_respects_current_yaw_without_changing_input():
    q = np.zeros(43)
    q[3:7] = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    original = q.copy()
    np.testing.assert_allclose(
        world_command_to_body(q, (0.3, 0.0, 0.0)), [0.0, -0.3, 0.0], atol=1e-7
    )
    np.testing.assert_array_equal(q, original)
    with pytest.raises(ValueError):
        world_command_to_body(q, (True, 0.0, 0.0))
