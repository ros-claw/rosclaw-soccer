import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from test_shared_native_motor_world import source_xml

from rosclaw_soccer.rsi.contact_reference_ik import (
    first_contact_foot,
    solve_absolute_contact_pose,
    solve_contact_reference_pose,
)
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene


def fixture(ball_x):
    tree = ET.fromstring(source_xml())
    tree.find(".//body[@name='ball']").set("pos", f"{ball_x} 0 1")
    for joint in tree.findall(".//joint"):
        joint.set("limited", "true")
        joint.set("range", "-2 2")
    for side, knee, foot in (("left", 3, 5), ("right", 9, 11)):
        body = tree.find(f".//body[@name='link{knee}']")
        body.set("name", side + "_knee_link")
        body.find("geom").set("name", side + "_shin")
        body.find("geom").set("pos", "0 -0.2 0")
        body = tree.find(f".//body[@name='link{foot}']")
        body.set("name", side + "_ankle_roll_link")
        body.find("geom").set("contype", "1")
        body.find("geom").set("conaffinity", "1")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(ET.tostring(tree, encoding="unicode"), prefixes=("a/",))
    )
    data = mujoco.MjData(model)
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="ball")
    return model, data, layout


@pytest.mark.parametrize(
    "ball_x,expected",
    [
        (0.12, "KINEMATIC_CONTACT_CONFIGURATION_FEASIBLE"),
        (1.0, "ABSTAIN_LOCAL_CONTACT_OBJECTIVE_TOO_FAR"),
    ],
)
def test_offline_geometry_never_steps_or_changes_source_root_ball_or_joints(ball_x, expected):
    model, data, layout = fixture(ball_x)
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    result = solve_contact_reference_pose(model, data, layout, goal=np.array([8.0, 0.0, 0.11]))
    assert result["status"] == expected
    for observed, original in zip((data.qpos, data.qvel, data.ctrl), before[:3], strict=True):
        np.testing.assert_array_equal(observed, original)
    assert data.time == before[3] == 0
    assert result["new_physical_steps"] == 0
    assert not result["physically_qualified"] and not result["hardware_authorized"]


@pytest.mark.parametrize(
    "goal", [np.array([np.nan, 0.0, 0.0]), np.zeros(2), np.zeros(3, np.float32)]
)
def test_malformed_goal_rejected_before_solver(goal):
    model, data, layout = fixture(0.12)
    with pytest.raises(ValueError):
        solve_contact_reference_pose(model, data, layout, goal=goal)


def test_explicit_contact_foot_is_not_replaced_by_closer_other_foot():
    model, data, layout = fixture(0.12)
    result = solve_contact_reference_pose(
        model, data, layout, goal=np.array([8.0, 0.0, 0.11]), active_foot="right"
    )
    assert result["active_foot"] == "right"
    assert result["contact_foot_selection"] == "explicit"
    with pytest.raises(ValueError, match="contact foot"):
        solve_contact_reference_pose(
            model, data, layout, goal=np.array([8.0, 0.0, 0.11]), active_foot="auto"
        )


def test_first_native_foot_is_not_largest_later_impact():
    forces = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 40.0]])
    assert first_contact_foot(forces) == ("left", 1)
    assert first_contact_foot(np.zeros((10, 2))) is None
    with pytest.raises(ValueError):
        first_contact_foot(np.array([[np.nan, 0.0]]))


def test_absolute_geometry_has_distinct_authority_and_no_physical_side_effects():
    model, data, layout = fixture(0.12)
    before = data.qpos.copy()
    result = solve_absolute_contact_pose(
        model, data, layout, goal=np.array([8.0, 0.0, 0.11]), active_foot="right"
    )
    assert result["status"] == "KINEMATIC_CONTACT_CONFIGURATION_FEASIBLE"
    assert result["schema"] == "soccer.absolute_contact_reference_geometry.v1"
    assert not result["original_035_total_displacement_certificate_claimed"]
    assert result["physical_joint_limits_enforced"]
    assert result["per_iteration_step_cap_rad"] == 0.1
    assert result["new_neural_calls"] == result["new_physical_steps"] == 0
    np.testing.assert_array_equal(data.qpos, before)
    with pytest.raises(ValueError, match="explicit active foot"):
        solve_absolute_contact_pose(
            model, data, layout, goal=np.array([8.0, 0.0, 0.11]), active_foot="auto"
        )
