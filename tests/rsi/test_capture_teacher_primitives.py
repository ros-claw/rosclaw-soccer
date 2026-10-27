import xml.etree.ElementTree as ET
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from test_paid_foot_taskspace import synthetic_episode

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.measured_foot_capture_teacher import (
    CaptureTeacherConfig,
    MeasuredFootCaptureTeacher,
)
from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent


@pytest.mark.parametrize(
    "field,value",
    [
        ("anticipation_s", float("nan")),
        ("toe_up_rad", float("inf")),
        ("maximum_joint_delta_rad", 0.21),
        ("delta_slew_rad_per_control", 0.1),
        ("contact_front_m", 0.19),
        ("foot_height_below_ball_m", -0.1),
        ("shin_clearance", 1),
        ("clearance_profile", "unknown"),
        ("bilateral_clearance", 1),
    ],
)
def test_capture_teacher_rejects_unbounded_or_nonfinite_declarations(field, value):
    with pytest.raises(ValueError):
        replace(CaptureTeacherConfig(), **{field: value}).validate()


def test_predictive_clearance_requires_enabled_shin_constraint():
    CaptureTeacherConfig(clearance_profile="anticipatory-priority-v1").validate()
    with pytest.raises(ValueError):
        CaptureTeacherConfig(
            shin_clearance=False, clearance_profile="anticipatory-priority-v1"
        ).validate()
    CaptureTeacherConfig(bilateral_clearance=True).validate()
    with pytest.raises(ValueError):
        CaptureTeacherConfig(shin_clearance=False, bilateral_clearance=True).validate()


def test_prefixed_kinematic_teacher_keeps_HG_order_and_never_writes_live_state(tmp_path):
    source = synthetic_episode(with_ball=True)
    path = tmp_path / "source.xml"
    mujoco.mj_saveLastXML(str(path), source.env.model)
    tree = ET.parse(path)
    robot = tree.getroot().find("./worldbody/body[@name='pelvis']")
    for elem in robot.iter():
        if "name" in elem.attrib:
            elem.set("name", "robot/" + elem.attrib["name"])
    m = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    qi = np.array([m.jnt_qposadr[m.joint("robot/" + n).id] for n in G1_DDS_JOINT_NAMES])
    ep = SimpleNamespace(
        frame=100,
        env=SimpleNamespace(model=m, data=d, pelvis_body_id=m.body("robot/pelvis").id),
        layout=SimpleNamespace(
            qi=qi, ball=m.body("ball").id, canonical=lambda data: (np.zeros(43), np.zeros(41))
        ),
        offset=np.zeros(29, np.float32),
        scale=np.ones(29, np.float32),
        order=np.arange(29),
    )
    before = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.xpos.copy(), m.geom_size.copy()
    solver = FootTaskSpaceIntent(
        m, foot="left", center=105, latent=np.zeros((2, 6)), robot_prefix="robot/"
    )
    action, plan = solver.solve_offset(
        ep,
        np.zeros(29, np.float32),
        np.array([0.02, 0, 0.02, 0, 0, 0]),
        shin_clearance=True,
        observed_shin_clearance=True,
    )
    assert np.isfinite(action).all() and np.abs(plan["joint_delta_HG"]).max() <= 0.2 + 1e-12
    assert plan["ik_iterations"] == 4
    for a, b in zip(before, (d.qpos, d.qvel, d.ctrl, d.xpos, m.geom_size), strict=True):
        np.testing.assert_array_equal(a, b)
    assert d.time == 0


def test_capture_teacher_identity_rate_limit_and_executed_action_history(monkeypatch):
    import rosclaw_soccer.rsi.measured_foot_capture_teacher as module

    class Solver:
        def __init__(self, model, *, foot, **kwargs):
            self.body = 1 if foot == "left" else 2
            self.kin = SimpleNamespace(
                qpos=np.zeros(43),
                xpos=np.array([[0, 0, 0.8], [0, 0.1, 0.05], [0, -0.1, 0.05], [0.4, 0.1, 0.11]]),
                xmat=np.tile(np.eye(3).ravel(), (4, 1)),
            )

        def solve_offset(self, ep, nominal, offset, **kwargs):
            delta = np.zeros(29)
            delta[:6] = 0.2
            return nominal.copy(), dict(joint_delta_HG=delta, ik_iterations=4)

    monkeypatch.setattr(module, "FootTaskSpaceIntent", Solver)
    monkeypatch.setattr(mujoco, "mj_kinematics", lambda *args: None)
    monkeypatch.setattr(mujoco, "mj_comPos", lambda *args: None)
    v = np.zeros(41)
    layout = SimpleNamespace(
        pelvis=0, ball=3, qi=np.arange(7, 36), canonical=lambda d: (np.zeros(43), v.copy())
    )
    d = SimpleNamespace(
        xpos=np.array([[0.0, 0.0, 0.8], [0.0, 0.1, 0.05], [0.0, -0.1, 0.05], [5.0, 0.0, 0.11]]),
        xmat=np.tile(np.eye(3).ravel(), (4, 1)),
        qpos=np.zeros(43),
    )
    world = SimpleNamespace(
        model=None, data=d, layouts={"b/": layout}, control_steps=0, failed=False
    )
    params = SimpleNamespace(
        action_offset=np.full(29, 0.1, np.float32), action_scale=np.full(29, 0.5, np.float32)
    )
    teacher = MeasuredFootCaptureTeacher(world, "b/", params, config=CaptureTeacherConfig())
    nominal = np.linspace(-0.1, 0.1, 29, dtype=np.float32)
    baseline = (nominal * params.action_scale + params.action_offset).astype(np.float64)
    target, trace = teacher.propose(nominal, previous_forces=np.zeros(3))
    np.testing.assert_array_equal(target, baseline)
    assert not trace["teacher_active"] and trace["teacher_ik_iterations"] == 0
    with pytest.raises(ValueError, match="consecutive"):
        teacher.propose(nominal, previous_forces=np.zeros(3))
    world.control_steps = 1
    d.xpos[3] = [0.4, 0.1, 0.11]
    v[35] = -2
    before = d.qpos.copy(), d.xpos.copy()
    target, trace = teacher.propose(nominal, previous_forces=np.zeros(3))
    assert trace["teacher_active"] and trace["teacher_foot"] == 0
    assert abs(trace["teacher_delta_HG"]).max() == pytest.approx(0.03)
    assert trace["teacher_is_neural_action"] is False
    expected = (
        trace["teacher_applied_action"] * params.action_scale + params.action_offset
    ).astype(np.float64)
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(nominal, np.linspace(-0.1, 0.1, 29, dtype=np.float32))
    for x, y in zip(before, (d.qpos, d.xpos), strict=True):
        np.testing.assert_array_equal(x, y)
    bilateral = MeasuredFootCaptureTeacher(
        world,
        "b/",
        params,
        config=CaptureTeacherConfig(
            bilateral_clearance=True, clearance_profile="anticipatory-priority-v1"
        ),
    )
    _, bilateral_trace = bilateral.propose(nominal, previous_forces=np.zeros(3))
    assert bilateral_trace["teacher_opposite_ik_iterations"] == 4
    assert bilateral_trace["teacher_ik_iterations"] == 8
    assert abs(bilateral_trace["teacher_delta_HG"]).max() == pytest.approx(0.03)
    for x, y in zip(before, (d.qpos, d.xpos), strict=True):
        np.testing.assert_array_equal(x, y)
