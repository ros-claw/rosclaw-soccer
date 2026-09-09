from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def effect(monkeypatch, progress, **changes):
    mujoco = pytest.importorskip("mujoco")

    def jac(model, data, jp, jr, point, body):
        jp[:, :3] = np.eye(3)

    monkeypatch.setattr(mujoco, "mj_jac", jac)
    kwargs = dict(
        model=SimpleNamespace(nv=29),
        data=SimpleNamespace(xpos=np.array([[0.0, 0.0, 0.1]]), qvel=np.zeros(29)),
        ankle_body_id=0,
        actuated_dof_indices=np.arange(29),
        ball_position_m=np.array([0.2, 0.0, 0.115]),
        ball_velocity_mps=np.array([1.0, 0.0, 0.0]),
        desired_ball_direction_xy=np.array([1.0, 0.0]),
        contact_mode="receive",
        local_lateral_sign=1.0,
        contact_recent=True,
        config=G1LocomotionContactTeacherConfig(),
        receive_capture_progress=progress,
    )
    kwargs.update(changes)
    return locomotion_contact_teacher_effect(**kwargs)


def test_capture_slows_foot_without_strike_followthrough_or_ball_writes(monkeypatch):
    ball = np.array([0.2, 0.0, 0.115])
    velocity = np.array([1.0, 0.0, 0.0])
    a = effect(monkeypatch, 0.0, ball_position_m=ball, ball_velocity_mps=velocity)
    b = effect(monkeypatch, 1.0, ball_position_m=ball, ball_velocity_mps=velocity)
    np.testing.assert_array_equal(ball, [0.2, 0.0, 0.115])
    np.testing.assert_array_equal(velocity, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(a.ankle_target_m, b.ankle_target_m)
    assert a.ankle_target_m[0] == pytest.approx(0.14)
    assert a.task_force_n[0] - b.task_force_n[0] == pytest.approx(7.0)
    assert np.max(np.abs(b.torque_nm)) <= 14
    old = effect(monkeypatch, None)
    assert old.ankle_target_m[0] == pytest.approx(0.34)


@pytest.mark.parametrize("progress", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_capture_requires_bounded_explicit_progress(monkeypatch, progress):
    with pytest.raises(ValueError, match="capture progress"):
        effect(monkeypatch, progress)


@pytest.mark.parametrize(
    "changes", [dict(contact_recent=False), dict(contact_mode="strike"), dict(strike_progress=0.5)]
)
def test_capture_cannot_be_used_as_an_unobserved_or_strike_event(monkeypatch, changes):
    with pytest.raises(ValueError, match="capture progress"):
        effect(monkeypatch, 0.5, **changes)


def test_default_config_hash_and_strict_handoff_boundary():
    old = IndependentTeamWorldConfig()
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    with pytest.raises(ValueError):
        replace(old, post_receive_contact_control=True)
    with pytest.raises(ValueError):
        replace(old, strict_receive_handoff=True, post_receive_contact_control=1)
    assert (
        replace(old, strict_receive_handoff=True, post_receive_contact_control=True).config_hash
        != old.config_hash
    )


def test_committed_offset_rejects_runtime_incompatible_range_before_simulation():
    with pytest.raises(ValueError, match="SIM-only"):
        G1LocomotionContactTeacherConfig(committed_receive_ankle_lateral_offset_m=0.1)
    good = G1LocomotionContactTeacherConfig(committed_receive_ankle_lateral_offset_m=0.12)
    replace(good, receive_ankle_lateral_offset_m=good.committed_receive_ankle_lateral_offset_m)
