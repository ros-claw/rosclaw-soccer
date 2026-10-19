"""Synthetic collision heading arithmetic, no physical outcome claims."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.physics.initial_contact_heading import (
    heading_deg,
    initial_contact_heading,
    reconstruct_contact_orientation,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def fixture(first=1200):
    arrays = dict(
        canonical_qpos_before_substep=np.zeros((3000, 43)),
        canonical_qvel_after_substep=np.zeros((3000, 41)),
        robot_contact_force_on_ball_world_n=np.zeros((3000, 3)),
    )
    row = dict(
        case_id="synthetic",
        source_receipt_hash="source",
        original_outcome=dict(task_pass=False),
        first_foot_contact_event=None,
        robot_contact_events_over_1n=[],
    )
    if first is not None:
        event = dict(
            substep_index=first,
            other_body_name="left_ankle_roll_link",
            force_norm_n=10.0,
            other_body_origin_world_m=[0.0, 0.0, 0.0],
            normal_other_to_ball_world=[1.0, 0.0, 0.0],
            force_on_ball_world_n=[3.0, 3.0, 0.0],
        )
        row["first_foot_contact_event"] = event
        row["robot_contact_events_over_1n"] = [event]
        arrays["robot_contact_force_on_ball_world_n"][first:] = [3.0, 3.0, 0.0]
        arrays["canonical_qvel_after_substep"][first:, 35:38] = [1.0, 1.0, 0.0]
        arrays["canonical_qpos_before_substep"][:, 36:39] = [1.0, 2.0, 3.0]
    return row, arrays


@pytest.mark.parametrize("first", [None, 0, 1200, 2995, 2999])
def test_keeps_nofoot_and_short_diagnostic_windows_without_changing_m0(first):
    row, arrays = fixture(first)
    before = copy.deepcopy(row)
    r = initial_contact_heading(row, arrays)
    assert r["original_outcome"] == dict(task_pass=False)
    assert r["new_physical_steps"] == r["optimizer_updates"] == 0
    assert r["headings_not_new_task_gates"]
    if first is None:
        assert r["contact"] is None
    else:
        c = r["contact"]
        duration = min(10, 3000 - first) * 0.002
        assert c["initial_window_duration_s"] == duration
        assert c["initial20ms_fully_observed"] is (first <= 2990)
        assert c["robot_impulse_initial_window_heading_deg"] == pytest.approx(45)
        assert c["ball_velocity_after_initial_window_heading_deg"] == pytest.approx(45)
        assert c["first_force_heading_deg"] == pytest.approx(45)
        assert c["first_normal_heading_deg"] == 0.0
        np.testing.assert_allclose(
            c["robot_impulse_initial_window_world_n_s"], [3 * duration, 3 * duration, 0]
        )
    assert before == row


@pytest.mark.parametrize(
    "mutation",
    [
        "short",
        "nan",
        "dtype",
        "index",
        "truthy",
        "weak",
        "unowned",
        "normal",
        "nofoot",
        "not_first",
    ],
)
def test_malformed_arrays_or_ambiguous_first_contact_are_not_interpretable(mutation):
    row, arrays = fixture()
    event = row["first_foot_contact_event"]
    if mutation == "short":
        arrays["canonical_qvel_after_substep"] = arrays["canonical_qvel_after_substep"][:-1]
    elif mutation == "nan":
        arrays["canonical_qvel_after_substep"][0, 0] = np.nan
    elif mutation == "dtype":
        arrays["canonical_qvel_after_substep"] = arrays["canonical_qvel_after_substep"].astype(
            np.float32
        )
    elif mutation == "index":
        event["substep_index"] = 3000
    elif mutation == "truthy":
        event["substep_index"] = True
    elif mutation == "weak":
        event["force_norm_n"] = 1.0
    elif mutation == "unowned":
        row["robot_contact_events_over_1n"] = []
    elif mutation == "normal":
        event["normal_other_to_ball_world"] = [2.0, 0.0, 0.0]
    elif mutation == "nofoot":
        row["first_foot_contact_event"] = None
    else:
        row["robot_contact_events_over_1n"].append(dict(event, substep_index=10))
    with pytest.raises(ValueError):
        initial_contact_heading(row, arrays)


def test_undefined_planar_heading_is_unknown_not_zero():
    assert heading_deg([0.0, 0.0, 1.0]) is None
    assert heading_deg([-1.0, 0.0, 0.0]) == 180
    with pytest.raises(ValueError):
        heading_deg([float("nan"), 0.0, 0.0])


def kinematic_fixture():
    import mujoco

    bodies, motors = [], []
    for i, name in enumerate(G1_DDS_JOINT_NAMES):
        body = (
            "left_ankle_roll_link"
            if i == 5
            else "right_ankle_roll_link"
            if i == 11
            else "body_" + name
        )
        bodies.append(
            f'<body name="{body}" pos=".1 0 .1">'
            f'<joint name="{name}" type="hinge" axis="0 0 1" range="-2 2"/>'
            '<geom type="sphere" size=".01" mass=".01"/></body>'
        )
        motors.append(f'<motor joint="{name}"/>')
    xml = (
        '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size=".05" mass="1"/>'
        + "".join(bodies)
        + '</body><body name="ball" pos="1 2 3"><freejoint/>'
        '<geom type="sphere" size=".11" mass=".43"/></body></worldbody><actuator>'
        + "".join(motors)
        + "</actuator></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12)]
    mujoco.mj_kinematics(model, data)
    row, arrays = fixture()
    event = row["first_foot_contact_event"]
    event["other_body_origin_world_m"] = data.xpos[model.body("left_ankle_roll_link").id].tolist()
    event["normal_other_to_ball_world"] = [np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0]
    arrays["canonical_qpos_before_substep"][:] = data.qpos
    return model, row, arrays


def test_collision_stage_fk_proves_orientation_not_a_dynamic_counterfactual(monkeypatch):
    import mujoco

    model, row, arrays = kinematic_fixture()
    monkeypatch.setattr(mujoco, "mj_step", lambda *a: pytest.fail("no physics allowed"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *a: pytest.fail("no forward dynamics needed"))
    r = reconstruct_contact_orientation(model, row, arrays)
    assert r["foot_forward_heading_world_deg"] == pytest.approx(30)
    assert r["pelvis_forward_heading_world_deg"] == pytest.approx(30)
    np.testing.assert_allclose(r["normal_in_foot_body_frame"], [1, 0, 0], atol=1e-15)
    assert r["foot_origin_maximum_reconstruction_error_m"] == 0
    assert r["joint_angles_not_intervened"]
    assert not r["dynamic_counterfactual_performed"]
    assert r["new_physical_steps"] == 0


@pytest.mark.parametrize("mutation", ["origin", "root_quaternion", "ball_quaternion"])
def test_fk_rejects_wrong_collision_clock_or_invalid_canonical_pose(mutation):
    model, row, arrays = kinematic_fixture()
    if mutation == "origin":
        row["first_foot_contact_event"]["other_body_origin_world_m"][0] += 0.01
    else:
        indices = slice(3, 7) if mutation == "root_quaternion" else slice(39, 43)
        arrays["canonical_qpos_before_substep"][:, indices] = 0
    with pytest.raises(ValueError):
        reconstruct_contact_orientation(model, row, arrays)


def test_nofoot_does_not_fabricate_an_orientation_or_allocate_a_state():
    row, arrays = fixture(None)
    assert reconstruct_contact_orientation(None, row, arrays) is None
