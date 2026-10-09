"""Independent course/state checks, not private exam admission tests."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import rosclaw_soccer.rsi.cpu_motor_transfer_evidence as audit_module
from rosclaw_soccer.rsi.native_first_touch_initial_state import NativeFirstTouchInitialState


def fixture():
    course = (2.5, 0.04, -0.6)
    joints = np.linspace(-0.2, 0.2, 29, dtype=np.float64)
    qpos = np.zeros(43, dtype=np.float64)
    qvel = np.zeros(41, dtype=np.float64)
    qpos[:7] = [0, 0, 0.793, 1, 0, 0, 0]
    qpos[7:36] = joints
    qpos[36:] = [2.5, 0.04, 0.13, 1, 0, 0, 0]
    qvel[35:] = [-0.6, 0, 0, 0, -0.6 / 0.11, 0]
    return course, joints, qpos, qvel


def test_exact_full_initial_state_and_private_contract():
    course, joints, qpos, qvel = fixture()
    reference = NativeFirstTouchInitialState(course, joints)
    reference.verify(list(course), qpos, qvel)
    contract = reference.contract()
    assert contract["initial_state_checked_exactly"]
    for key in (
        "course_freshness_certified",
        "exam_admission_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert contract[key] is False
    assert "2.5" not in repr(reference)
    assert "2.5" not in json.dumps(contract)
    assert all(
        value.startswith("sha256:") for key, value in contract.items() if key.endswith("hash")
    )


@pytest.mark.parametrize("field,width", [("qpos", 43), ("qvel", 41)])
def test_every_state_coordinate_is_checked(field, width):
    course, joints, qpos, qvel = fixture()
    reference = NativeFirstTouchInitialState(course, joints)
    for index in range(width):
        position, velocity = qpos.copy(), qvel.copy()
        changed = position if field == "qpos" else velocity
        changed[index] = np.nextafter(changed[index], np.inf)
        with pytest.raises(ValueError, match="initial canonical"):
            reference.verify(list(course), position, velocity)


def test_independent_snapshot_owns_original_defaults():
    course, joints, qpos, qvel = fixture()
    reference = NativeFirstTouchInitialState(course, joints)
    before = reference.contract()
    joints[:] = 999
    reference.verify(list(course), qpos, qvel)
    assert reference.contract() == before
    with pytest.raises(FrozenInstanceError):
        reference._qpos_bytes = b"changed"


@pytest.mark.parametrize("vx", [-0.6, 0.0, 0.6])
def test_rolling_direction_not_sliding_or_spinning_oppositely(vx):
    course, joints, qpos, qvel = fixture()
    course = (course[0], course[1], vx)
    qvel[35], qvel[39] = vx, vx / 0.11
    reference = NativeFirstTouchInitialState(course, joints)
    reference.verify(list(course), qpos, qvel)
    if vx:
        for angular in (0, -vx / 0.11, vx / 0.1):
            bad = qvel.copy()
            bad[39] = angular
            with pytest.raises(ValueError, match="initial canonical velocity"):
                reference.verify(list(course), qpos, bad)


@pytest.mark.parametrize(
    "value", [True, np.float64(0.1), float("nan"), float("inf"), "0.1", None, 10**1000]
)
def test_bad_course_value_rejected_without_disclosing_coordinates(value):
    course, joints, qpos, qvel = fixture()
    with pytest.raises(ValueError, match="declared course"):
        NativeFirstTouchInitialState((course[0], value, course[2]), joints)
    reference = NativeFirstTouchInitialState(course, joints)
    with pytest.raises(ValueError, match="declared course"):
        reference.verify([course[0], value, course[2]], qpos, qvel)


@pytest.mark.parametrize("bad", [[], [2.5, 0.04, -0.6], (2.5, 0.04), None])
def test_constructor_requires_exact_three_tuple(bad):
    _, joints, _, _ = fixture()
    with pytest.raises(ValueError, match="declared course"):
        NativeFirstTouchInitialState(bad, joints)


@pytest.mark.parametrize("kind", ["float32", "int", "bool", "shape", "nan", "list"])
def test_bad_joint_defaults_rejected(kind):
    course, joints, _, _ = fixture()
    if kind in ("float32", "int", "bool"):
        joints = joints.astype(kind)
    elif kind == "shape":
        joints = joints.reshape(1, 29)
    elif kind == "nan":
        joints[0] = np.nan
    else:
        joints = joints.tolist()
    with pytest.raises(ValueError, match="float64"):
        NativeFirstTouchInitialState(course, joints)


@pytest.mark.parametrize("field", ["qpos", "qvel"])
@pytest.mark.parametrize("kind", ["float32", "shape", "nan", "inf", "list"])
def test_malformed_trace_vectors_rejected(field, kind):
    course, joints, qpos, qvel = fixture()
    reference = NativeFirstTouchInitialState(course, joints)
    vector = qpos if field == "qpos" else qvel
    if kind == "float32":
        vector = vector.astype(np.float32)
    elif kind == "shape":
        vector = vector.reshape(1, -1)
    elif kind in ("nan", "inf"):
        vector[0] = float(kind)
    else:
        vector = vector.tolist()
    with pytest.raises(ValueError, match="float64"):
        reference.verify(
            list(course), vector if field == "qpos" else qpos, vector if field == "qvel" else qvel
        )


@pytest.mark.parametrize("index", range(3))
def test_report_course_cannot_be_relabelled(index):
    course, joints, qpos, qvel = fixture()
    reference = NativeFirstTouchInitialState(course, joints)
    changed = list(course)
    changed[index] = np.nextafter(changed[index], np.inf).item()
    with pytest.raises(ValueError, match="independent declaration"):
        reference.verify(changed, qpos, qvel)


@pytest.mark.parametrize("impostor", [True, {}, object()])
def test_audit_rejects_fake_reference_before_read_or_physics(tmp_path, monkeypatch, impostor):
    def unopened(*args, **kwargs):
        raise AssertionError("invalid reference must reject before evidence IO")

    monkeypatch.setattr(audit_module, "_sealed", unopened)
    with pytest.raises(ValueError, match="exact independent native"):
        audit_module.audit_cpu_transfer(
            tmp_path, tmp_path / "source.py", initial_state_reference=impostor
        )
