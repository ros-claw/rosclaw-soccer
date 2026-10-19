"""Toy topology/target callback physics only, NOT complete G1 feedback qualification."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecaster
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.feedback_branch_control import G1FeedbackBranchControl
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes


def world():
    names = {
        3: "left_knee_link",
        4: "left_ankle_pitch_link",
        5: "left_ankle_roll_link",
        9: "right_knee_link",
        10: "right_ankle_pitch_link",
        11: "right_ankle_roll_link",
    }
    bodies = "".join(
        f'<body name="{names.get(i, "body_" + str(i))}" pos="{i * 0.15} 0 .25">'
        f'<joint name="{n}" axis="0 1 0"/>'
        '<geom type="sphere" size=".02" mass=".2"/></body>'
        for i, n in enumerate(G1_DDS_JOINT_NAMES)
    )
    actuators = "".join(f'<motor joint="{n}"/>' for n in G1_DDS_JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002" integrator="Euler"/><worldbody>'
        '<geom type="plane" size="0 0 .01"/><body name="pelvis" pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".04" mass="5" contype="0" conaffinity="0"/>'
        + bodies
        + '</body><body name="ball" pos="5 0 3"><freejoint/>'
        '<geom name="ball_geom" type="sphere" size=".11" mass=".43"/>'
        "</body></worldbody><actuator>" + actuators + "</actuator></mujoco>"
    )
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    data = mujoco.MjData(model)
    data.time = 0.6
    mujoco.mj_forward(model, data)
    forecaster = G1FeedbackForecaster(
        model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        max_forecast_steps=40,
    )
    return model, data, forecaster


def control(model):
    result = object.__new__(G1FeedbackBranchControl)
    result.model, result.frame, result.consumed = model, 30, False
    return result


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_actual_updated_target_at20ms_and_full_solver_cache_not_refreshed(monkeypatch, offset):
    model, data, f = world()
    seen = []

    def next_target(self, d, forces):
        assert d is not data and d.model is model and d.time == pytest.approx(0.62)
        assert not np.array_equal(d.xpos[model.body("pelvis").id], d.qpos[:3])
        assert forces.shape == (6,)
        seen.append(d.qpos.copy())
        self.consumed = True
        return np.full(29, 0.2), dict(joint_target_rad=np.full(29, 0.2))

    monkeypatch.setattr(G1FeedbackBranchControl, "next_target", next_target)
    before = data.qpos.copy()
    prediction = f.forecast(
        data, np.zeros(29), control=control(model), first_offset_nm=np.full(29, offset)
    )
    assert prediction.arrays["nonfoot_robot_contact_over1n_after_substep"].dtype == np.bool_
    assert prediction.arrays["foot_contact_peak_force_n_after_substep"].shape == (20, 2)
    assert prediction.arrays["completed_foot_ground_normal_n"].shape == (20, 2)
    assert np.all(prediction.arrays["completed_foot_ground_normal_n"] >= 0)
    actual = mujoco.MjData(model)
    mujoco.mj_copyData(actual, model, data)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    for step in range(20):
        target = np.zeros(29) if step < 10 else np.full(29, 0.2)
        raw = (target - actual.qpos[7:36]) * 10 - actual.qvel[6:35] * 0.2
        requested = raw + offset if step < 10 and offset else raw
        actual.ctrl[:] = np.clip(requested, -limits, limits)
        mujoco.mj_step(model, actual)
        np.testing.assert_array_equal(
            actual.qpos, prediction.arrays["canonical_qpos_after_substep"][step]
        )
        np.testing.assert_array_equal(
            actual.qvel, prediction.arrays["canonical_qvel_after_substep"][step]
        )
        np.testing.assert_array_equal(actual.ctrl, prediction.arrays["applied_control_nm"][step])
    np.testing.assert_array_equal(data.qpos, before)
    assert data.time == 0.6 and len(seen) == 1
    again = f.forecast(
        data, np.zeros(29), control=control(model), first_offset_nm=np.full(29, offset)
    )
    assert again.end_integration_state.tobytes() == prediction.end_integration_state.tobytes()
    assert f.attempted_steps == f.integrated_steps == 40
    assert f.next_target_attempts == f.next_target_completed == 2
    with pytest.raises(RuntimeError, match="budget"):
        f.forecast(data, np.zeros(29), control=control(model), first_offset_nm=np.zeros(29))


def test_failed_next_target_charges_only_actual_first_ten_steps(monkeypatch):
    model, data, f = world()

    def fail(self, *args):
        self.consumed = True
        raise ValueError("synthetic branch failure")

    monkeypatch.setattr(G1FeedbackBranchControl, "next_target", fail)
    with pytest.raises(ValueError, match="synthetic"):
        f.forecast(data, np.zeros(29), control=control(model), first_offset_nm=np.zeros(29))
    assert f.attempted_steps == f.integrated_steps == 10
    assert f.next_target_attempts == 1 and f.next_target_completed == 0 and data.time == 0.6
