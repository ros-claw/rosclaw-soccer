"""Synthetic reference kinematics only, never G1 dynamic qualification."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.rsi.contact_reference_offset import (
    bounded_point_delta,
    offset_contact_reference,
)


def reference():
    q = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, np.zeros(29)]
    return SonicPoseReference(
        tuple(tuple(float(x) for x in q) for _ in range(251)), "sha256:" + "1" * 64
    )


def test_zero_is_exact_identity_and_no_kinematic_allocation(monkeypatch):
    def forbidden(*args):
        raise AssertionError("zero must allocate no MuJoCo data")

    monkeypatch.setattr(mujoco, "MjData", forbidden)
    original = reference()
    result, report = offset_contact_reference(None, original, offset_root_m=(0.0, 0.0, 0.0))
    assert result is original and report["zero_is_bit_exact_identity"]


@pytest.mark.parametrize(
    "offset", [[0.0, 0.0, 0.0], (0.05, 0.0, 0.0), (np.nan, 0.0, 0.0), (0.04, 0.04, 0.04)]
)
def test_invalid_request_rejected_before_allocation(offset):
    with pytest.raises(ValueError, match="bounded immutable"):
        offset_contact_reference(None, reference(), offset_root_m=offset)


def test_dls_direction_and_native_reference_limits():
    jac = np.c_[np.eye(3), np.zeros((3, 3))]
    limits = np.tile([-0.1, 0.1], (6, 1))
    result = bounded_point_delta(
        jac, np.array([0.02, -0.03, 0.0]), np.zeros(6), np.zeros(6), limits
    )
    assert result[0] > 0 > result[1] and abs(result).max() <= 0.04
    current = np.full(6, 0.1)
    result = bounded_point_delta(jac, np.array([0.1, 0.0, 0.0]), current, np.zeros(6), limits)
    assert result.max() <= 0.1 and result.min() >= -0.1


def synthetic_model():
    right = G1_DDS_JOINT_NAMES[6:12]
    chain = ""
    for i, (name, axis) in enumerate(
        zip(right, ("0 1 0", "1 0 0", "0 0 1", "0 1 0", "0 1 0", "1 0 0"), strict=True)
    ):
        body = "right_ankle_roll_link" if i == 5 else "leg" + str(i)
        chain += (
            f'<body name="{body}" pos="0 0 -.1">'
            f'<joint name="{name}" axis="{axis}" range="-2 2"/>'
            '<geom type="box" size=".02 .02 .02" mass="1"/>'
        )
    for x in (-0.04, 0.12):
        for y in (-0.03, 0.03):
            chain += f'<geom type="sphere" pos="{x} {y} -.04" size=".005"/>'
    chain += "</body>" * 6
    rest = "".join(
        f'<body pos="0 0 .1"><joint name="{n}" range="-2 2"/><geom size=".01" mass=".1"/></body>'
        for n in G1_DDS_JOINT_NAMES
        if n not in right
    )
    actuators = "".join(f'<motor joint="{n}"/>' for n in G1_DDS_JOINT_NAMES)
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option timestep=".002"/>'
        '<default><joint limited="true"/></default>'
        '<worldbody><body name="pelvis"><freejoint/><geom size=".1" mass="5"/>'
        + chain
        + rest
        + '</body><body name="ball"><freejoint/><geom size=".11" mass=".43"/></body></worldbody>'
        "<actuator>" + actuators + "</actuator></mujoco>"
    )


def test_synthetic_fk_changes_only_right_reference_leg_and_tapers(monkeypatch):
    model = synthetic_model()
    original = reference()

    def forbidden(*args, **kwargs):
        raise AssertionError("reference adaptation cannot integrate physics")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    modified, report = offset_contact_reference(model, original, offset_root_m=(0.0, 0.015, -0.015))
    a, b = np.asarray(original.poses), np.asarray(modified.poses)
    np.testing.assert_array_equal(a[:91], b[:91])
    np.testing.assert_array_equal(a[210:], b[210:])
    np.testing.assert_array_equal(a[:, :13], b[:, :13])
    np.testing.assert_array_equal(a[:, 19:], b[:, 19:])
    assert np.any(a[:, 13:19] != b[:, 13:19])
    assert report["maximum_reference_delta_rad"] <= 0.12000000001
    assert report["maximum_fk_residual_m"] < 0.005
    assert report["new_native_steps"] == 0 and not report["physics_qualified"]
    modified.initialize(a[0], frames=410)


def test_wrong_model_rejected_before_data_allocation():
    model = SimpleNamespace(nq=3, nv=3, nu=1)
    with pytest.raises(ValueError, match="original G1"):
        offset_contact_reference(model, reference(), offset_root_m=(0.0, 0.0, -0.02))
