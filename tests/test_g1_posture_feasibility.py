"""Static checks must not turn a valid-looking pose into motion authority."""

import dataclasses

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.posture_feasibility import audit_g1_static_posture

mujoco = pytest.importorskip("mujoco")


def model_fixture(*, prefix: str = "", overlap: bool = False, gear: float = 1.0):
    bodies = []
    motors = []
    for index, name in enumerate(G1_DDS_JOINT_NAMES):
        x = 0.5 if overlap and index < 2 else 0.5 + index * 0.25
        bodies.append(
            f'<body name="link{index}" pos="{x} 0 0">'
            f'<joint name="{prefix}{name}" type="hinge" axis="0 1 0" range="-1 1"/>'
            f'<geom name="sphere{index}" pos=".1 0 0" type="sphere" size=".1" mass=".1"/>'
            "</body>"
        )
        motors.append(f'<motor joint="{prefix}{name}" gear="{gear}"/>')
    xml = (
        '<mujoco><compiler angle="radian"/>'
        '<worldbody><geom type="plane" size="10 10 .1"/>'
        f'<body name="{prefix}pelvis" pos="0 0 2"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        + "".join(bodies)
        + "</body></worldbody><actuator>"
        + "".join(motors)
        + "</actuator></mujoco>"
    )
    return mujoco.MjModel.from_xml_string(xml)


def test_floating_pose_is_not_an_equilibrium_and_input_is_untouched():
    model = model_fixture()
    qpos = model.qpos0.copy()
    original = qpos.copy()
    result = audit_g1_static_posture(model, qpos)
    np.testing.assert_array_equal(qpos, original)
    assert "unactuated_root_force" in result.rejection_reasons
    assert not result.reference_feasible
    assert result.activation_ceiling == "SIM_ONLY"
    assert not result.promotion_eligible
    assert len(result.inverse_joint_torque_nm) == 29
    assert len(result.unactuated_root_wrench) == 6
    assert not result.self_contacts
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.promotion_eligible = True


def test_native_self_collision_is_not_hidden_by_joint_or_torque_clipping():
    model = model_fixture(overlap=True)
    result = audit_g1_static_posture(model, model.qpos0)
    assert "robot_self_penetration" in result.rejection_reasons
    assert any(contact.signed_distance_m < 0 for contact in result.self_contacts)
    assert not result.reference_feasible


def test_joint_violation_is_reported_not_projected_away():
    model = model_fixture()
    qpos = model.qpos0.copy()
    qpos[7] = 1.1
    result = audit_g1_static_posture(model, qpos)
    assert result.maximum_joint_violation_rad == pytest.approx(0.1)
    assert "joint_limit_violation" in result.rejection_reasons
    assert qpos[7] == 1.1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pose_rejects(value):
    model = model_fixture()
    qpos = model.qpos0.copy()
    qpos[0] = value
    with pytest.raises(ValueError, match="finite native"):
        audit_g1_static_posture(model, qpos)


def test_malformed_pose_and_quaternion_reject():
    model = model_fixture()
    with pytest.raises(ValueError, match="finite native"):
        audit_g1_static_posture(model, [0.0])
    qpos = model.qpos0.copy()
    qpos[3:7] = 0.0
    with pytest.raises(ValueError, match="normalized"):
        audit_g1_static_posture(model, qpos)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_residual_ceilings_reject(value):
    model = model_fixture()
    with pytest.raises(ValueError, match="ceilings"):
        audit_g1_static_posture(model, model.qpos0, maximum_root_force_n=value)
    with pytest.raises(ValueError, match="ceilings"):
        audit_g1_static_posture(model, model.qpos0, maximum_root_moment_nm=value)


def test_explicit_role_prefix_and_missing_contract():
    model = model_fixture(prefix="goalkeeper_")
    assert not audit_g1_static_posture(model, model.qpos0, prefix="goalkeeper_").reference_feasible
    with pytest.raises(ValueError, match="missing"):
        audit_g1_static_posture(model, model.qpos0)


def test_unsupported_motor_does_not_guess_inverse_torque_mapping():
    model = model_fixture(gear=2.0)
    with pytest.raises(ValueError, match="unit-gear"):
        audit_g1_static_posture(model, model.qpos0)


def test_model_motor_limits_are_not_replaced_by_larger_hard_limits():
    model = model_fixture()
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:] = (-0.01, 0.01)
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:] = (-0.01, 0.01)
    result = audit_g1_static_posture(model, model.qpos0)
    assert "inverse_torque_exceeds_model_control_limit" in result.rejection_reasons
    assert "inverse_torque_exceeds_model_force_limit" in result.rejection_reasons


def test_audit_constructor_cannot_request_promotion():
    result = audit_g1_static_posture(model_fixture(), model_fixture().qpos0)
    with pytest.raises(ValueError, match="init=False"):
        dataclasses.replace(result, promotion_eligible=True)
