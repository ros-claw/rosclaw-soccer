import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_foot_observation import (
    first_contact_control_entry,
    measure_foot_ball_kinematics,
)
from scripts.rsi_audit_relay_foot_entry import finite_difference


def model_fixture():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="ball" pos="0.6 0 0.2"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
      <body name="left" pos="0 0.1 0.1"><joint name="l" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05" mass="1"/></body>
      <body name="right" pos="0 -0.1 0.1"><joint name="r" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05" mass="1"/></body>
    </worldbody></mujoco>""")


def test_scratch_kinematics_match_finite_difference_without_mutating_state():
    model = model_fixture()
    q, v = model.qpos0.copy(), np.array([0.5, -0.1, 0.2, 0, 0, 0.3, 0.7, -0.4])
    oldq, oldv = q.copy(), v.copy()
    kwargs = dict(foot_body_names=("left", "right"), ball_body_name="ball", ball_dofadr=0)
    result = measure_foot_ball_kinematics(model, q, v, **kwargs)
    future = q.copy()
    mujoco.mj_integratePos(model, future, v, 1e-6)
    next_result = measure_foot_ball_kinematics(model, future, v, **kwargs)
    assert np.allclose((next_result[:, :3] - result[:, :3]) / 1e-6, result[:, 3:], atol=1e-6)
    assert np.array_equal(q, oldq) and np.array_equal(v, oldv)
    with pytest.raises(ValueError):
        measure_foot_ball_kinematics(model, q, v, **(kwargs | {"ball_dofadr": 1}))


def test_event_selection_uses_previous_control_not_future_substep():
    q = np.zeros((4, 10, 79))
    v = np.zeros((4, 10, 76))
    forces = np.zeros((4, 10, 2, 3))
    q[1, -1, 0] = 1
    q[2, :, 0] = 99
    forces[2, 3, 1, 0] = 3
    result = first_contact_control_entry(q, v, forces, np.zeros(79), np.zeros(76))
    assert result is not None
    state, _, control, substep, categories = result
    assert (state[0], control, substep, categories) == (1, 2, 3, (0,))
    state[0] = 7
    assert q[1, -1, 0] == 1
    forces[:] = 0
    assert first_contact_control_entry(q, v, forces, np.zeros(79), np.zeros(76)) is None


def test_independent_finite_difference_auditor():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="soccer_ball" pos="0.6 0 0.2"><freejoint/>
        <geom type="sphere" size="0.1" mass="1"/></body>
      <body name="b/left_ankle_roll_link" pos="0 0.1 0.1">
        <joint type="hinge" axis="0 0 1"/><geom type="sphere" size="0.05" mass="1"/></body>
      <body name="b/right_ankle_roll_link" pos="0 -0.1 0.1">
        <joint type="hinge" axis="0 0 1"/><geom type="sphere" size="0.05" mass="1"/></body>
    </worldbody></mujoco>""")
    q, v = model.qpos0.copy(), np.array([0.5, -0.1, 0.2, 0, 0, 0.3, 0.7, -0.4])
    actual = measure_foot_ball_kinematics(
        model,
        q,
        v,
        foot_body_names=("b/left_ankle_roll_link", "b/right_ankle_roll_link"),
        ball_body_name="soccer_ball",
        ball_dofadr=0,
    )
    assert np.allclose(finite_difference(model, q, v), actual, rtol=0, atol=1e-6)
