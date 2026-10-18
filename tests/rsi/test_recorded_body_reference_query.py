"""Synthetic canonical model, zero dynamics, explicit IK stub for phase tests."""

import copy

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi import recorded_body_reference_query as query
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from tests.rsi.test_contact_reference_labels import trace as label_trace


def fixture():
    bodies = []
    names = {
        3: "left_knee_link",
        5: "left_ankle_roll_link",
        9: "right_knee_link",
        11: "right_ankle_roll_link",
    }
    for i, joint in enumerate(G1_DDS_JOINT_NAMES):
        z = -0.65 if i == 5 else -0.72
        bodies.append(
            f'<body name="{names.get(i, "body" + str(i))}" pos="0 0 {z}">'
            f'<joint name="{joint}" type="hinge" range="-3.5 3.5"/>'
            '<geom type="sphere" size="0.02" mass="1"/></body>'
        )
    actuators = "".join(f'<motor joint="{n}"/>' for n in G1_DDS_JOINT_NAMES)
    xml = (
        '<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="Euler"/>'
        '<worldbody><body name="pelvis" pos="0 0 0.8"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="5"/>' + "".join(bodies) + "</body>"
        '<body name="ball" pos="0.5 0 0.11"><freejoint/>'
        '<geom name="ball_geom" type="sphere" size="0.11" mass="0.43"/>'
        "</body></worldbody><actuator>" + actuators + "</actuator></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    qi = model.jnt_qposadr[[model.joint(n).id for n in G1_DDS_JOINT_NAMES]]
    ball = model.body("ball").id
    bq = model.jnt_qposadr[model.body_jntadr[ball]]
    canonical = np.concatenate((data.qpos[:7], data.qpos[qi], data.qpos[bq : bq + 7]))
    trace = label_trace()
    trace.update(
        canonical_qpos=np.tile(canonical, (300, 1, 1)),
        canonical_qvel=np.zeros((300, 1, 41)),
        root_pose_xyzw_m=np.tile([0, 0, 0.8, 0, 0, 0, 1.0], (300, 1, 1)),
        ball_position_before_step_m=np.tile(data.xpos[ball], (300, 1, 1)),
        foot_geometry_position_before_step_m=np.tile(
            data.xpos[
                [
                    model.body(n).id
                    for n in (
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_knee_link",
                        "right_knee_link",
                    )
                ]
            ],
            (300, 1, 1, 1),
        ),
        force_n=np.zeros((300, 1, 6)),
    )
    dense = dict(canonical_qpos_before_substep=np.tile(canonical, (3000, 1)))
    return trace, dense, model


def test_scratch_query_never_integrates_or_runs_forward_dynamics(monkeypatch):
    trace, dense, model = fixture()
    original = copy.deepcopy(trace)

    def forbidden(*args, **kw):
        raise AssertionError("no dynamics permitted")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    result = query.query_recorded_body(trace, dense, model, control=BoundedFirstTouchControl())
    assert result["proposed_reference_increment_rad"].shape == (300, 12)
    assert not result["proposed_reference_increment_rad"][:30].any()
    assert np.max(np.abs(result["proposed_reference_increment_rad"])) <= 0.35
    assert result["diagnostic"]["kinematic_cache_frames_verified"] == 300
    assert not result["diagnostic"]["dataset_admission_authorized"]
    for k in trace:
        np.testing.assert_array_equal(trace[k], original[k])


def test_completed_contact_only_changes_next_label_and_release_is_causal(monkeypatch):
    trace, dense, model = fixture()
    monkeypatch.setattr(query, "continuous_swing_joint_delta", lambda *a, **kw: np.full(6, 0.1))
    a = query.query_recorded_body(trace, dense, model, control=BoundedFirstTouchControl())
    trace["force_n"][40, 0, 0] = 2
    trace["future_success"] = "not an actor or expert input"
    b = query.query_recorded_body(trace, dense, model, control=BoundedFirstTouchControl())
    np.testing.assert_array_equal(
        a["proposed_reference_increment_rad"][:41], b["proposed_reference_increment_rad"][:41]
    )
    np.testing.assert_allclose(b["proposed_reference_increment_rad"][41, :6], 0.095)
    assert not b["proposed_reference_increment_rad"][60:].any()
    assert b["diagnostic"]["first_recorded_contact_frame"] == 40


@pytest.mark.parametrize("tamper", ["shape", "nan", "dtype", "clock", "cached", "force", "model"])
def test_rejects_missing_or_wrong_measured_clock(tamper):
    trace, dense, model = fixture()
    if tamper == "shape":
        trace["canonical_qpos"] = trace["canonical_qpos"][:299]
    elif tamper == "nan":
        dense["canonical_qpos_before_substep"][0, 0] = np.nan
    elif tamper == "dtype":
        dense["canonical_qpos_before_substep"] = dense["canonical_qpos_before_substep"].astype(
            np.float32
        )
    elif tamper == "clock":
        dense["canonical_qpos_before_substep"][10, 0] += 0.1
    elif tamper == "cached":
        dense["canonical_qpos_before_substep"][9, 0] += 0.1
    elif tamper == "force":
        trace["force_n"][299, 0, 0] = -1
    else:
        model.opt.timestep = 0.003
    with pytest.raises(ValueError):
        query.query_recorded_body(trace, dense, model, control=BoundedFirstTouchControl())
