import json

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.contact_dynamics import read_ground_contact_dynamics


def world(tilt=0.0):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option cone="elliptic"/>'
        '<worldbody><geom name="floor" type="plane" size="2 2 .1" '
        f'euler="0 {tilt} 0"/>'
        '<body name="foot" pos="0 0 .09"><freejoint name="floating"/>'
        '<geom name="foot_geom" size=".1" mass="1"/></body>'
        '<body name="arm" pos="1 0 1"><joint name="hinge" type="hinge" '
        'axis="0 1 0" limited="true" range="-.1 .1"/>'
        '<geom name="arm_geom" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    data.qvel[:2] = [1, 0.4]
    mujoco.mj_forward(model, data)
    return model, data


def read(model, data, **kwargs):
    arguments = dict(
        ground_geom_id=int(model.geom("floor").id),
        joint_names=("hinge",),
        support_body_names=("foot",),
    )
    return read_ground_contact_dynamics(model, data, **(arguments | kwargs))


@pytest.mark.parametrize("tilt", [0.0, 0.3, -0.3])
def test_force_is_on_robot_not_on_ground_and_joint_units_are_explicit(tilt):
    model, data = world(tilt)
    result = read(model, data)
    np.testing.assert_allclose(
        result["support_body_ground_force_world_n"]["foot"], data.qfrc_constraint[:3], atol=1e-10
    )
    joint = int(model.jnt_dofadr[model.joint("hinge").id])
    assert result["joint_generalized_constraint_torque_nm"] == [data.qfrc_constraint[joint]]
    assert result["ground_contacts"][0]["normal_force_n"] > 0
    assert result["ground_contacts"][0]["other_body_name"] == "foot"
    assert result["support_readiness_or_balance_qualified"] is False
    json.dumps(result, allow_nan=False)


def test_read_does_not_step_refresh_change_warmstart_or_alias_native_arrays():
    model, data = world()
    mujoco.mj_step(model, data)
    fields = (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "qfrc_constraint",
        "qfrc_actuator",
        "ctrl",
        "xpos",
    )
    before = {name: getattr(data, name).copy() for name in fields}
    clock = data.time
    result = read(model, data)
    for name in fields:
        np.testing.assert_array_equal(getattr(data, name), before[name])
    assert data.time == clock
    assert result["data_time_sec"] == clock
    assert result["kinematics_refreshed"] is False
    assert result["controls_mutated"] is False
    result["recorded_joint_velocity_rad_s"][0] = 1234
    np.testing.assert_array_equal(data.qvel, before["qvel"])


def test_missing_ground_contact_is_observed_zero_not_a_successful_support_claim():
    model, data = world()
    data.qpos[2] = 0.5
    mujoco.mj_forward(model, data)
    result = read(model, data)
    assert result["ground_contacts"] == []
    assert result["support_body_ground_force_world_n"]["foot"] == [0, 0, 0]
    assert result["support_readiness_or_balance_qualified"] is False


def test_zero_force_contact_is_retained():
    model, data = world()
    data.efc_force[:] = 0
    result = read(model, data)
    assert len(result["ground_contacts"]) == 1
    assert result["ground_contacts"][0]["normal_force_n"] == 0


def test_joint_limit_constraint_is_identified_but_constraint_torque_not_called_ground_only():
    model, data = world()
    data.joint("hinge").qpos[0] = 0.2
    mujoco.mj_forward(model, data)
    result = read(model, data)
    assert result["active_joint_limit_constraint_names"] == ["hinge"]
    assert result["joint_generalized_constraint_torque_nm"][0] != 0
    assert result["constraint_torque_is_not_ground_only"] is True


@pytest.mark.parametrize("ground", [True, -1, 1000, 1.0])
def test_invalid_ground_ids_rejected(ground):
    model, data = world()
    with pytest.raises(ValueError, match="world-plane"):
        read(model, data, ground_geom_id=ground)


def test_robot_geometry_is_not_accepted_as_ground():
    model, data = world()
    with pytest.raises(ValueError, match="world-plane"):
        read(model, data, ground_geom_id=int(model.geom("foot_geom").id))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_names": []},
        {"joint_names": ()},
        {"joint_names": ("hinge", "hinge")},
        {"joint_names": (True,)},
        {"support_body_names": ("foot", "foot")},
        {"support_body_names": ()},
    ],
)
def test_names_must_be_explicit_unique_and_bounded(kwargs):
    model, data = world()
    with pytest.raises(ValueError, match="distinct explicit"):
        read(model, data, **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"joint_names": ("missing",)}, {"support_body_names": ("missing",)}]
)
def test_missing_model_elements_refused(kwargs):
    model, data = world()
    with pytest.raises(ValueError, match="element missing"):
        read(model, data, **kwargs)


def test_free_joint_cannot_be_reported_in_hinge_torque_units():
    model, data = world()
    with pytest.raises(ValueError, match="hinge joints"):
        read(model, data, joint_names=("floating",))


def test_world_cannot_be_robot_support():
    model, data = world()
    with pytest.raises(ValueError, match="world body"):
        read(model, data, support_body_names=("world",))


def test_same_shaped_other_model_data_refused():
    model, _ = world()
    _, data = world()
    with pytest.raises(ValueError, match="exact supplied model"):
        read(model, data)


@pytest.mark.parametrize("field", ["qfrc_constraint", "qacc", "qvel"])
def test_nonfinite_dynamics_rejected(field):
    model, data = world()
    getattr(data, field)[-1] = np.nan
    with pytest.raises(ValueError, match="solved contact dynamics"):
        read(model, data)
