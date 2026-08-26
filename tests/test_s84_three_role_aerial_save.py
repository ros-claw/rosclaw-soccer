from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    _balanced_dive_phase_profile,
    _goalkeeper_current_relative_intercept_y,
    _goalkeeper_punch_force_local,
    _landing_capture_profile,
    _latch_gmt_mirror_direction,
    _mirror_gmt_proprioception,
    _prestrike_positioning_target,
)
from rosclaw_soccer.training.three_role_aerial_save import (
    ThreeRoleAerialSaveConfig,
    evaluate_three_role_aerial_save,
    three_role_aerial_save_kwargs,
)


def _empty_result() -> G1SharedWorldResult:
    return G1SharedWorldResult(
        finite_state=True,
        pass_contact_observed=True,
        shot_contact_observed=False,
        pass_contact_time_sec=1.0,
        shot_contact_time_sec=None,
        pass_peak_ball_speed_mps=2.0,
        shot_peak_ball_speed_mps=0.0,
        goal_crossed=False,
        goal_plane_crossed=False,
        goal_crossing_y_m=None,
        goal_crossing_z_m=None,
        target_error_m=None,
        passer_min_pelvis_height_m=0.75,
        shooter_min_pelvis_height_m=0.75,
        passer_roll_peak_rad=0.0,
        passer_pitch_peak_rad=0.0,
        shooter_roll_peak_rad=0.0,
        shooter_pitch_peak_rad=0.0,
        passer_tail_wobble_index=0.0,
        shooter_tail_wobble_index=0.0,
        receiver_phase_hold_frames=0,
        receiver_phase_advance_frames=0,
        receiver_max_ball_phase_error_m=0.0,
        robot_robot_contact_count=0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        physics_steps=1,
    )


def test_three_role_aerial_save_config_is_sim_only_and_fail_closed() -> None:
    config = ThreeRoleAerialSaveConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    assert config.minimum_pass_precision_m == pytest.approx(0.05)
    assert config.minimum_hand_height_m == pytest.approx(1.20)
    assert config.goalkeeper_support_arm_overhead_bias_rad == pytest.approx(0.26)
    assert config.goalkeeper_glove_contact_damping_ratio == pytest.approx(0.15)
    assert config.maximum_post_contact_speed_mps == pytest.approx(10.0)
    assert config.maximum_glove_surface_separation_m == pytest.approx(0.001)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="5 cm"):
        replace(config, minimum_pass_precision_m=0.051)
    with pytest.raises(ValueError, match="support-arm bias"):
        replace(config, goalkeeper_support_arm_overhead_bias_rad=0.36)
    with pytest.raises(ValueError, match="deflection-speed ceiling"):
        replace(config, maximum_post_contact_speed_mps=5.9)
    with pytest.raises(ValueError, match="glove separation"):
        replace(config, maximum_glove_surface_separation_m=0.003)


def test_three_role_aerial_save_requires_bound_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="artifacts"):
        three_role_aerial_save_kwargs(
            striker_actor_path=missing,
            goalkeeper_actor_path=missing,
            gmt_model_path=missing,
            gmt_skill_path=missing,
        )


def test_goalkeeper_support_arm_posture_requires_bimanual_reach(tmp_path: Path) -> None:
    actor = tmp_path / "actor.json"
    actor.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="posture requires bimanual reach"):
        G1GoalkeeperConfig(
            actor_observation_mode="visible_ball_history_v3",
            actor_artifact_path=actor,
            actor_bimanual_support_arm_overhead_bias_rad=0.10,
        )
    with pytest.raises(ValueError, match="joint-group scale"):
        G1GoalkeeperConfig(balanced_dive_lower_body_scale=1.01)
    with pytest.raises(ValueError, match="activation lead"):
        G1GoalkeeperConfig(balanced_dive_activation_lead_sec=1.01)
    with pytest.raises(ValueError, match="vertical punch"):
        G1GoalkeeperConfig(actor_bimanual_punch_vertical_force_scale=1.01)
    with pytest.raises(ValueError, match="outward punch"):
        G1GoalkeeperConfig(actor_bimanual_punch_outward_force_scale=0.76)
    assert G1GoalkeeperConfig().mosaic_gmt_mirror_by_intercept is True
    assert G1GoalkeeperConfig().canonical_locomotion_mirror_enabled is False
    with pytest.raises(ValueError, match="mirror flag"):
        G1GoalkeeperConfig(mosaic_gmt_mirror_by_intercept="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="locomotion mirror flag"):
        G1GoalkeeperConfig(canonical_locomotion_mirror_enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="regulation positioning flag"):
        G1GoalkeeperConfig(regulation_goal_positioning_enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prestrike ball blend"):
        G1GoalkeeperConfig(prestrike_ball_lateral_blend=1.01)


def test_three_role_aerial_save_missing_contact_fails_closed() -> None:
    report = evaluate_three_role_aerial_save(
        result=_empty_result(),
        trajectory={
            "time": np.asarray((0.0,), dtype=np.float64),
            "ball_pose": np.zeros((1, 7), dtype=np.float64),
            "ball_velocity": np.zeros((1, 6), dtype=np.float64),
        },
        config=ThreeRoleAerialSaveConfig(),
    )
    assert report == {"passed": False, "reason": "missing ordered shot/glove contact"}


def test_three_role_aerial_save_non_finite_rollout_fails_closed() -> None:
    report = evaluate_three_role_aerial_save(
        result=replace(_empty_result(), finite_state=False),
        trajectory={
            "time": np.asarray((0.0,), dtype=np.float64),
            "ball_pose": np.zeros((1, 7), dtype=np.float64),
            "ball_velocity": np.zeros((1, 6), dtype=np.float64),
        },
        config=ThreeRoleAerialSaveConfig(),
    )
    assert report == {"passed": False, "reason": "non-finite rollout"}


def test_three_role_aerial_save_rejects_phantom_glove_contact() -> None:
    result = replace(
        _empty_result(),
        shot_contact_observed=True,
        shot_contact_time_sec=2.0,
        goalkeeper_enabled=True,
        goalkeeper_ball_contact_observed=True,
        goalkeeper_ball_contact_time_sec=3.0,
        goalkeeper_save_observed=True,
        goalkeeper_right_glove_contact_observed=True,
        goalkeeper_glove_contact_height_m=1.45,
        goalkeeper_glove_contact_time_sec=3.0,
        goalkeeper_glove_contact_position_m=(6.7, 0.2, 1.45),
        goalkeeper_glove_contact_surface_distance_m=0.010,
        goalkeeper_glove_contact_side="right",
        goalkeeper_contact_left_hand_height_m=1.30,
        goalkeeper_contact_right_hand_height_m=1.30,
        goalkeeper_both_hands_raised_at_contact=True,
        goalkeeper_min_pelvis_height_m=0.75,
        pass_delivery_error_m=0.01,
        pass_delivery_lateral_error_m=0.01,
    )
    time = np.arange(5, dtype=np.float64)
    pose = np.zeros((5, 7), dtype=np.float64)
    pose[:, 2] = (0.2, 0.4, 1.50, 1.45, 1.20)
    velocity = np.zeros((5, 6), dtype=np.float64)
    velocity[2, 0] = 9.0
    velocity[4, 0] = -4.0
    report = evaluate_three_role_aerial_save(
        result=result,
        trajectory={"time": time, "ball_pose": pose, "ball_velocity": velocity},
        config=ThreeRoleAerialSaveConfig(),
    )
    assert report["passed"] is False
    assert report["gates"]["anatomical_glove_contact"] is True
    assert report["gates"]["collision_faithful_glove_contact"] is False


def test_balanced_dive_recovery_handoff_is_continuous() -> None:
    settings = {
        "flight_duration_sec": 0.65,
        "phase_at_arrival": 0.60,
        "peak_phase": 1.0,
        "blend_in_sec": 0.24,
        "recovery_tail_sec": 0.40,
    }
    phase_duration = (0.65 - 0.24) / 0.60
    complete = 0.24 + phase_duration
    before = _balanced_dive_phase_profile(elapsed_sec=complete - 1.0e-6, **settings)
    at = _balanced_dive_phase_profile(elapsed_sec=complete, **settings)
    after = _balanced_dive_phase_profile(elapsed_sec=complete + 1.0e-6, **settings)

    assert before[0] < at[0] == after[0] == pytest.approx(1.0)
    assert before[1] == pytest.approx(1.0)
    assert at[1] == pytest.approx(1.0)
    assert after[1] == pytest.approx(1.0 - 1.0e-6 / 0.40)
    assert before[2] is at[2] is after[2] is True
    assert _balanced_dive_phase_profile(
        elapsed_sec=complete + 0.40,
        **settings,
    )[1:] == pytest.approx((0.0, True))

    capped = {**settings, "peak_phase": 0.65}
    capped_complete = 0.24 + 0.65 * phase_duration
    assert _balanced_dive_phase_profile(
        elapsed_sec=capped_complete,
        **capped,
    ) == pytest.approx((0.65, 1.0, True))
    assert _balanced_dive_phase_profile(
        elapsed_sec=capped_complete + 0.40,
        **capped,
    ) == pytest.approx((0.65, 0.0, True))
    assert _balanced_dive_phase_profile(
        elapsed_sec=capped_complete + 0.400001,
        **capped,
    )[2] is False

    preloaded = {**settings, "initial_phase": 0.20}
    preloaded_duration = (0.65 - 0.24) / (0.60 - 0.20)
    assert _balanced_dive_phase_profile(elapsed_sec=0.24, **preloaded)[0] == pytest.approx(0.20)
    assert _balanced_dive_phase_profile(
        elapsed_sec=0.24 + (0.60 - 0.20) * preloaded_duration,
        **preloaded,
    )[0] == pytest.approx(0.60)


def test_landing_capture_profile_is_continuous_and_bounded() -> None:
    assert _landing_capture_profile(elapsed_sec=0.0, duration_sec=0.8) == (0.0, True)
    assert _landing_capture_profile(elapsed_sec=0.4, duration_sec=0.8) == pytest.approx(
        (0.5, True)
    )
    assert _landing_capture_profile(elapsed_sec=0.8, duration_sec=0.8) == (1.0, True)
    assert _landing_capture_profile(elapsed_sec=0.81, duration_sec=0.8) == (1.0, False)
    with pytest.raises(ValueError, match="landing capture timing"):
        _landing_capture_profile(elapsed_sec=-0.1, duration_sec=0.8)


def test_gmt_mirror_direction_is_latched_for_one_skill_event() -> None:
    latched = _latch_gmt_mirror_direction(
        None,
        skill_active=True,
        local_intercept_y_m=0.30,
        mirror_enabled=True,
    )
    assert latched is True
    assert (
        _latch_gmt_mirror_direction(
            latched,
            skill_active=True,
            local_intercept_y_m=-0.20,
            mirror_enabled=True,
        )
        is True
    )
    assert (
        _latch_gmt_mirror_direction(
            None,
            skill_active=False,
            local_intercept_y_m=0.30,
            mirror_enabled=True,
        )
        is None
    )
    with pytest.raises(ValueError, match="finite"):
        _latch_gmt_mirror_direction(
            None,
            skill_active=True,
            local_intercept_y_m=float("nan"),
            mirror_enabled=True,
        )


def test_gmt_mirrors_inputs_and_outputs_in_the_same_half_space() -> None:
    position = np.arange(29, dtype=np.float64) / 100.0
    velocity = -position
    quaternion = np.asarray((0.90, 0.10, -0.20, 0.30), dtype=np.float64)
    angular_velocity = np.asarray((0.40, -0.50, 0.60), dtype=np.float64)

    expected = (position, velocity, quaternion, angular_velocity)
    mirrored = _mirror_gmt_proprioception(*expected)
    restored = _mirror_gmt_proprioception(*mirrored)

    assert all(
        np.allclose(actual, wanted)
        for actual, wanted in zip(restored, expected, strict=True)
    )
    assert mirrored[2] == pytest.approx((0.90, -0.10, -0.20, -0.30))
    assert mirrored[3] == pytest.approx((-0.40, -0.50, -0.60))


def test_goalkeeper_punch_force_is_outward_symmetric() -> None:
    right = _goalkeeper_punch_force_local(
        force_n=20.0,
        local_intercept_y_m=0.50,
        vertical_force_scale=0.40,
        outward_force_scale=0.25,
    )
    left = _goalkeeper_punch_force_local(
        force_n=20.0,
        local_intercept_y_m=-0.50,
        vertical_force_scale=0.40,
        outward_force_scale=0.25,
    )

    assert right == pytest.approx((20.0, 5.0, 8.0))
    assert left == pytest.approx((20.0, -5.0, 8.0))


def test_prestrike_positioning_is_visible_ball_conditioned_and_symmetric() -> None:
    right = _prestrike_positioning_target(
        ball_y_m=3.2,
        anchor_y_m=0.0,
        goal_width_m=7.32,
        blend=0.85,
    )
    left = _prestrike_positioning_target(
        ball_y_m=-3.2,
        anchor_y_m=0.0,
        goal_width_m=7.32,
        blend=0.85,
    )

    assert right == pytest.approx(2.72)
    assert left == pytest.approx(-2.72)
    assert _prestrike_positioning_target(
        ball_y_m=9.0,
        anchor_y_m=0.0,
        goal_width_m=7.32,
        blend=1.0,
    ) == pytest.approx(3.44)
    with pytest.raises(ValueError, match="finite"):
        _prestrike_positioning_target(
            ball_y_m=float("nan"),
            anchor_y_m=0.0,
            goal_width_m=7.32,
            blend=0.85,
        )


def test_goalkeeper_reach_gate_uses_current_body_not_spawn_anchor() -> None:
    right = SimpleNamespace(
        state=SimpleNamespace(pelvis_pos_w=np.asarray((0.0, -2.61, 0.0)))
    )
    left = SimpleNamespace(
        state=SimpleNamespace(pelvis_pos_w=np.asarray((0.0, 2.61, 0.0)))
    )

    assert _goalkeeper_current_relative_intercept_y(
        right,  # type: ignore[arg-type]
        anchor_relative_intercept_y_m=-3.35,
    ) == pytest.approx(-0.74)
    assert _goalkeeper_current_relative_intercept_y(
        left,  # type: ignore[arg-type]
        anchor_relative_intercept_y_m=3.35,
    ) == pytest.approx(0.74)
    with pytest.raises(ValueError, match="finite"):
        _goalkeeper_current_relative_intercept_y(
            right,  # type: ignore[arg-type]
            anchor_relative_intercept_y_m=float("nan"),
        )
