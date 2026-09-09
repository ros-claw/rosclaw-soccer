import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    contact_tracking_adjustment,
)
from rosclaw_soccer.training.team_contact_diagnostic import summarize


def test_opposing_controller_is_not_counted_as_a_completed_pass():
    residual = np.zeros((3, 29))
    residual[0, 0] = 10
    residual[1, 0] = 4
    trace = dict(
        contact_teacher_residual_torque_nm=residual,
        contact_teacher_baseline_torque_nm=-0.5 * residual,
        contact_teacher_guarded_torque_nm=0.5 * residual,
        contact_teacher_last_substep_valid=[True, True, False],
        ball_contact_force_n=[0, 0, 12],
        ball_contact_effector_code=[0, 0, 1],
        ball_contact_agent_code=[0, 0, 8],
    )
    r = summarize(trace)
    assert r["teacher_measured_frames"] == 2
    assert r["baseline_opposes_teacher_fraction"] == 1
    assert r["median_teacher_norm_nm"] == 7
    assert r["median_baseline_projection_nm"] == -3.5
    assert r["median_guarded_projection_nm"] == 3.5
    assert r["physical_foot_contact_frames"] == 1
    assert r["physical_foot_contact_agent_codes"] == [8]
    assert "completed_passes" not in r


@pytest.mark.parametrize("use_left", [True, False])
def test_tracking_adjustment_preserves_other_joints_and_default(use_left):
    torque = np.arange(29, dtype=float)
    original = torque.copy()
    np.testing.assert_array_equal(
        contact_tracking_adjustment(torque, use_left=use_left, scale=1), 0
    )
    adjustment = contact_tracking_adjustment(torque, use_left=use_left, scale=0.4)
    selected = np.arange(0, 6) if use_left else np.arange(6, 12)
    other = np.setdiff1d(np.arange(29), selected)
    np.testing.assert_allclose(adjustment[selected], -0.6 * torque[selected])
    np.testing.assert_array_equal(adjustment[other], 0)
    np.testing.assert_array_equal(torque, original)
    for invalid in (float("nan"), 0.39, 1.01):
        with pytest.raises(ValueError):
            contact_tracking_adjustment(torque, use_left=use_left, scale=invalid)
        with pytest.raises(ValueError):
            G1LocomotionContactTeacherConfig(contact_leg_stiffness_scale=invalid)


def test_joint_guard_margin_cannot_be_weakened_and_preserves_default_identity():
    from dataclasses import asdict

    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig

    default = IndependentTeamWorldConfig()
    legacy = asdict(default)
    for field in (
        "joint_guard_margin_rad",
        "keeper_reach",
        "glove_material",
        "strike_residual_enabled",
        "strike_stance_lateral_m",
        "motor_approach_standoff_m",
        "motor_idle_residual_fallback",
        "receiver_commitment_priority",
        "pass_stance_bypass",
        "receive_lateral_braking",
        "locomotion_action_frame_sync",
    ):
        legacy.pop(field)
    assert default.config_hash == str(hash_json(legacy))
    assert (
        IndependentTeamWorldConfig(joint_guard_margin_rad=0.08).config_hash != default.config_hash
    )
    for invalid in (0.039, 0.101, float("nan")):
        with pytest.raises(ValueError):
            IndependentTeamWorldConfig(joint_guard_margin_rad=invalid)


def test_control_profile_keeps_contact_and_safety_contracts_separate():
    from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
    from rosclaw_soccer.training.contact_control_profile import ContactControlProfile

    original = IndependentTeamWorldConfig()
    teacher = G1LocomotionContactTeacherConfig()
    world, motor = ContactControlProfile().apply(original, teacher)
    assert world.minimum_player_separation_m == 0.85
    assert world.joint_guard_margin_rad == 0.08
    assert world.minimum_pelvis_height_m == original.minimum_pelvis_height_m
    assert world.maximum_tilt_rad == original.maximum_tilt_rad
    assert motor.contact_leg_stiffness_scale == 0.8
    assert motor.maximum_joint_residual_nm == teacher.maximum_joint_residual_nm
    assert teacher.contact_leg_stiffness_scale == 1
    for change in (dict(activation_ceiling="REAL"), dict(guard_margin_rad=0.01)):
        with pytest.raises(ValueError):
            ContactControlProfile(**change)


def test_role_training_and_exams_receive_same_opt_in_profile(tmp_path, monkeypatch):
    from rosclaw_soccer.training import near_ball_curriculum as curriculum
    from rosclaw_soccer.training.contact_control_profile import ContactControlProfile

    calls = []
    monkeypatch.setattr(curriculum, "run_probe", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(curriculum.NearBallResidualPolicy, "load", lambda _: "numeric-policy")
    profile = ContactControlProfile()
    for explore in (True, False):
        job = curriculum.RoleRolloutJob(
            tmp_path,
            tmp_path / str(explore),
            tmp_path / "policy.npz",
            15,
            curriculum.RoleCourse("goalkeeper", False, 0.04),
            3,
            explore,
            True,
            profile,
        )
        curriculum.collect_role_course(job)
    assert all(c["contact_control_profile"] is profile for c in calls)
    assert all(c["basic_ball_play"] and c["strict_receive_handoff"] for c in calls)
    assert [c["near_ball_explore"] for c in calls] == [True, False]


def test_joint_safety_reward_is_agent_local_and_rejects_missing_evidence():
    from rosclaw_soccer.training.football_reward_shaping import joint_safety_penalty

    margins = np.full((3, 8, 29), 0.1)
    np.testing.assert_array_equal(joint_safety_penalty(margins), 0)
    margins[1, 4, 5] = -0.001
    penalty = joint_safety_penalty(margins)
    assert penalty[1, 4] == -0.27
    assert np.count_nonzero(penalty) == 1
    for invalid in (margins[:, :, :28], np.full((3, 8, 29), np.nan)):
        with pytest.raises(ValueError):
            joint_safety_penalty(invalid)


def test_audit_binds_declared_control_profile_to_physical_parameters():
    from rosclaw_soccer.training.contact_control_profile import ContactControlProfile
    from rosclaw_soccer.training.near_ball_learning_audit import _verify_contact_profile

    report = dict(
        world_config=dict(minimum_player_separation_m=0.85, joint_guard_margin_rad=0.08),
        contact_teacher_config=dict(contact_leg_stiffness_scale=0.8),
    )
    _verify_contact_profile(report, ContactControlProfile())
    report["world_config"]["joint_guard_margin_rad"] = 0.04
    with pytest.raises(ValueError, match="committed contact"):
        _verify_contact_profile(report, ContactControlProfile())


@pytest.mark.parametrize("epochs", [0, 17, True, 4.0])
def test_training_rejects_unbounded_or_noninteger_epochs_before_collection(tmp_path, epochs):
    from rosclaw_soccer.training.near_ball_residual_ppo import train

    with pytest.raises(ValueError, match="bounded online"):
        train(
            assets=tmp_path,
            output=tmp_path / "out",
            iterations=1,
            duration=5,
            optimizer_epochs=epochs,
        )
