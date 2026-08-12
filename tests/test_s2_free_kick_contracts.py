from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.asset_qualification import (
    G1AssetQualification,
    trajectory_digest,
)
from rosclaw_soccer.providers.g1.learned_runup import G1LearnedRunupConfig
from rosclaw_soccer.providers.g1.mujoco_primitives import build_shared_recovery_controller
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupConfig
from rosclaw_soccer.skills.shoot.free_kick import (
    G1FreeKickFlowConfig,
    _apply_compliant_net_force,
    _deepest_goal_mouth_point,
    _net_capture_plane_x,
    _select_contextual_phase,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def test_migrated_contract_schemas_remain_artifact_compatible() -> None:
    assert G1FreeKickFlowConfig().schema_version == (
        "rosclaw.simforge.g1_free_kick_flow_config.v36"
    )
    assert G1TrainingGoalSpec().schema_version == ("rosclaw.simforge.g1_training_goal_spec.v8")
    assert G1LearnedRunupConfig().schema_version == ("rosclaw.simforge.g1_learned_runup_config.v2")
    assert G1SonicRunupConfig().schema_version == ("rosclaw.simforge.g1_sonic_runup_config.v1")


def test_training_goal_uses_regulation_geometry_and_ball_dimensions() -> None:
    goal = G1TrainingGoalSpec(
        plane_x_m=8.5,
        width_m=7.32,
        height_m=2.44,
        depth_m=2.0,
        post_radius_m=0.05,
        target_y_m=3.1,
        target_z_m=2.1,
        precision_radius_m=0.10,
        ball_radius_m=0.69 / (2.0 * np.pi),
        ball_mass_kg=0.43,
        regulation_field_enabled=True,
    )

    assert 2.0 * np.pi * goal.ball_radius_m == pytest.approx(0.69)
    assert goal.ball_mass_kg == pytest.approx(0.43)
    assert goal.ball_angular_damping_n_m_s_rad == pytest.approx(0.00002)
    assert (goal.field_length_m, goal.field_width_m) == (105.0, 68.0)
    assert (goal.goal_area_depth_m, goal.penalty_area_depth_m) == (5.5, 16.5)
    assert goal.target_corner == "left_upper"
    assert goal.target_corner_center_m == pytest.approx((8.5, 3.5501830893, 2.3301830893))


def test_free_kick_configs_fail_closed_at_authority_boundaries() -> None:
    assert G1FreeKickFlowConfig(shot_foot_pitch_offset_rad=0.18)
    with pytest.raises(ValueError, match="shot foot pitch offset"):
        G1FreeKickFlowConfig(shot_foot_pitch_offset_rad=0.181)
    with pytest.raises(ValueError, match="shot COM shift"):
        G1FreeKickFlowConfig(shot_com_shift_y_m=-0.081)
    with pytest.raises(ValueError, match="lead duration"):
        G1FreeKickFlowConfig(ballistic_contact_lead_duration_sec=0.079)
    with pytest.raises(ValueError, match="recovery flag"):
        G1FreeKickFlowConfig(shared_cerebellar_recovery_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer multiple"):
        G1LearnedRunupConfig(control_dt_sec=0.019)
    with pytest.raises(ValueError, match="SONIC joint gain scales"):
        G1SonicRunupConfig(joint_gain_scales=(1.0,) * 28)


def test_contextual_phase_uses_proprioception_not_planner_identity() -> None:
    flow = G1FreeKickFlowConfig(
        kick_phase_start_frame=214,
        contextual_phase_yaw_threshold_rad=0.15,
        contextual_high_yaw_kick_phase_start_frame=190,
    )

    assert _select_contextual_phase(flow, 0.02) == (214, False)
    assert _select_contextual_phase(flow, 0.237) == (190, True)
    assert _select_contextual_phase(flow, -0.16) == (190, True)


def test_compliant_net_is_free_flight_then_dissipates_inside_pocket() -> None:
    goal = G1TrainingGoalSpec(plane_x_m=6.0)
    flow = G1FreeKickFlowConfig(net_capture_depth_m=0.20)
    data = SimpleNamespace(
        qpos=np.asarray((6.05, 0.0, 1.2), dtype=np.float64),
        qvel=np.asarray((8.0, 2.0, 3.0), dtype=np.float64),
        xfrc_applied=np.zeros((1, 6), dtype=np.float64),
    )
    ids = SimpleNamespace(ball=0, ball_qpos=0, ball_qvel=0)

    _apply_compliant_net_force(data, ids, goal, flow)
    np.testing.assert_allclose(data.xfrc_applied, 0.0)
    data.qpos[0] = 6.15
    _apply_compliant_net_force(data, ids, goal, flow)

    assert data.xfrc_applied[0, 0] < 0.0
    assert abs(data.xfrc_applied[0, 1]) < abs(data.xfrc_applied[0, 0]) * 0.1
    assert abs(data.xfrc_applied[0, 2]) < abs(data.xfrc_applied[0, 0]) * 0.1


def test_net_capture_geometry_and_deepest_point_are_physics_bound() -> None:
    goal = G1TrainingGoalSpec(depth_m=2.0)
    flow = G1FreeKickFlowConfig(net_capture_depth_m=1.6)
    low_contact = _net_capture_plane_x(goal, flow, goal.ball_radius_m)
    high_contact = _net_capture_plane_x(goal, flow, goal.height_m - goal.ball_radius_m)

    assert goal.plane_x_m < high_contact < low_contact < goal.plane_x_m + goal.depth_m
    deepest = _deepest_goal_mouth_point(None, np.asarray((5.04, 0.98, 0.115)), goal)
    deepest = _deepest_goal_mouth_point(deepest, np.asarray((5.16, 0.97, 0.116)), goal)
    assert _deepest_goal_mouth_point(deepest, np.asarray((5.12, 0.96, 0.115)), goal) == deepest


def test_trajectory_digest_is_dtype_shape_and_content_bound() -> None:
    trajectory = {
        "time": np.asarray((0.0, 0.1), dtype=np.float64),
        "qpos": np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64),
    }
    original = trajectory_digest(trajectory)

    assert trajectory_digest({key: value.copy() for key, value in trajectory.items()}) == original
    assert (
        trajectory_digest({**trajectory, "time": trajectory["time"].astype(np.float32)}) != original
    )
    assert trajectory_digest({**trajectory, "qpos": trajectory["qpos"] + 1e-12}) != original


def test_configs_are_json_ready_without_provider_objects() -> None:
    flow = G1FreeKickFlowConfig()
    goal = G1TrainingGoalSpec()

    assert asdict(flow)["approach_provider"] == "groot_history"
    assert asdict(goal)["ball_mass_kg"] == pytest.approx(0.41)


def test_recovery_bridge_preserves_frozen_shared_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import rosclaw_soccer.providers.g1.mujoco_primitives as primitives

    motion_dir = tmp_path / "policy/robonaldo/model"
    motion_dir.mkdir(parents=True)
    np.savez(motion_dir / "freekick_motion.npz", joint_pos=np.zeros((1, 29)))
    qualification = G1AssetQualification(
        eligible=True,
        asset_root=tmp_path,
        body_hash="sha256:" + "1" * 64,
        kick_prior_hash="sha256:" + "4" * 64,
        motion_hash="sha256:" + "2" * 64,
        backend_commit="0" * 40,
        actuator_count=29,
        joint_names=tuple(f"joint-{index}" for index in range(29)),
        policy_input_size=1190,
        policy_output_size=29,
        errors=(),
    )
    monkeypatch.setattr(
        primitives,
        "load_robonaldo",
        lambda _root: (object(), object(), np.arange(29), np.arange(29)),
    )

    controller = build_shared_recovery_controller(qualification)

    assert controller.config.start_policy_frame == 280
    assert controller.config.blend_frames == 80
    assert controller.config.standing_pose_blend == pytest.approx(0.02)
    assert controller.config.target_smoothing_alpha == pytest.approx(0.60)
    assert controller.config.target_smoothing_joint_group == "upper_body"
