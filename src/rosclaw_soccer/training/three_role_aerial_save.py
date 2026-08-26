"""Strict three-G1 pass, aerial strike and raised-hand save evidence loop.

The rollout remains one continuous CPU MuJoCo world with one physical ball.
Pixels are never used for scoring and the retained skill is SIM_ONLY.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.ballistic_contact_residual import (
    G1BallisticContactResidualConfig,
)
from rosclaw_soccer.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import (
    three_role_development_kwargs,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


@dataclass(frozen=True)
class ThreeRoleAerialSaveConfig:
    """Frozen SIM-only joint-skill parameters selected by failure replay."""

    simulation_duration_sec: float = 11.0
    shooter_start_sec: float = 1.96
    pass_reception_target_m: tuple[float, float, float] = (1.275, -0.020, 0.115)
    shooter_foot_yaw_offset_rad: float = 0.01
    shooter_foot_pitch_offset_rad: float = 0.12
    shooter_loft_synergy_rad: float = 0.15
    goalkeeper_depth_from_goal_line_m: float = 0.30
    goalkeeper_support_arm_blend: float = 0.40
    goalkeeper_support_arm_overhead_bias_rad: float = 0.26
    goalkeeper_reach_height_offset_m: float = -0.05
    goalkeeper_reach_local_x_m: float = 0.24
    goalkeeper_reach_gain_scale: float = 1.50
    goalkeeper_punch_force_n: float = 19.0
    goalkeeper_punch_central_boost_n_per_m: float = 38.0
    goalkeeper_punch_reference_abs_lateral_m: float = 0.45
    goalkeeper_glove_contact_time_constant_sec: float = 0.0075
    goalkeeper_glove_contact_damping_ratio: float = 0.15
    minimum_pass_precision_m: float = 0.05
    minimum_aerial_apex_m: float = 1.40
    minimum_glove_contact_height_m: float = 1.35
    minimum_hand_height_m: float = 1.20
    minimum_passer_pelvis_height_m: float = 0.60
    minimum_shooter_pelvis_height_m: float = 0.60
    minimum_goalkeeper_pelvis_height_m: float = 0.70
    minimum_incoming_speed_mps: float = 8.0
    minimum_outward_speed_mps: float = 3.0
    maximum_post_contact_speed_mps: float = 10.0
    maximum_glove_surface_separation_m: float = 0.001
    maximum_glove_surface_penetration_m: float = 0.018
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.three_role_aerial_save_config.v5"

    def __post_init__(self) -> None:
        values = (
            self.simulation_duration_sec,
            self.shooter_start_sec,
            *self.pass_reception_target_m,
            self.shooter_foot_yaw_offset_rad,
            self.shooter_foot_pitch_offset_rad,
            self.shooter_loft_synergy_rad,
            self.goalkeeper_depth_from_goal_line_m,
            self.goalkeeper_support_arm_blend,
            self.goalkeeper_support_arm_overhead_bias_rad,
            self.goalkeeper_reach_height_offset_m,
            self.goalkeeper_reach_local_x_m,
            self.goalkeeper_reach_gain_scale,
            self.goalkeeper_punch_force_n,
            self.goalkeeper_punch_central_boost_n_per_m,
            self.goalkeeper_punch_reference_abs_lateral_m,
            self.goalkeeper_glove_contact_time_constant_sec,
            self.goalkeeper_glove_contact_damping_ratio,
            self.minimum_pass_precision_m,
            self.minimum_aerial_apex_m,
            self.minimum_glove_contact_height_m,
            self.minimum_hand_height_m,
            self.minimum_passer_pelvis_height_m,
            self.minimum_shooter_pelvis_height_m,
            self.minimum_goalkeeper_pelvis_height_m,
            self.minimum_incoming_speed_mps,
            self.minimum_outward_speed_mps,
            self.maximum_post_contact_speed_mps,
            self.maximum_glove_surface_separation_m,
            self.maximum_glove_surface_penetration_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("three-role aerial-save config must be finite")
        if not 10.0 <= self.simulation_duration_sec <= 15.0:
            raise ValueError("three-role aerial-save duration is invalid")
        if not 1.90 <= self.shooter_start_sec <= 2.10:
            raise ValueError("three-role aerial-save shooter start is invalid")
        if not 0.0 <= self.goalkeeper_support_arm_blend <= 0.50:
            raise ValueError("three-role aerial-save support-arm blend is invalid")
        if not 0.0 <= self.goalkeeper_support_arm_overhead_bias_rad <= 0.35:
            raise ValueError("three-role aerial-save support-arm bias is invalid")
        if not -0.12 <= self.goalkeeper_reach_height_offset_m <= 0.12:
            raise ValueError("three-role aerial-save reach height is invalid")
        if not 0.0 <= self.goalkeeper_reach_local_x_m <= 0.35:
            raise ValueError("three-role aerial-save reach depth is invalid")
        if not 0.5 <= self.goalkeeper_reach_gain_scale <= 4.0:
            raise ValueError("three-role aerial-save reach gain is invalid")
        if not 0.0 <= self.goalkeeper_punch_force_n <= 80.0:
            raise ValueError("three-role aerial-save punch force is invalid")
        if not 0.0 <= self.goalkeeper_punch_central_boost_n_per_m <= 80.0:
            raise ValueError("three-role aerial-save central punch boost is invalid")
        if not 0.20 <= self.goalkeeper_punch_reference_abs_lateral_m <= 0.60:
            raise ValueError("three-role aerial-save punch reference is invalid")
        if not 0.0 < self.minimum_pass_precision_m <= 0.05:
            raise ValueError("three-role aerial-save pass gate cannot exceed 5 cm")
        if not 1.30 <= self.minimum_aerial_apex_m <= 2.20:
            raise ValueError("three-role aerial-save apex gate is not aerial")
        if not 1.25 <= self.minimum_glove_contact_height_m <= 2.0:
            raise ValueError("three-role aerial-save glove gate is too low")
        if not 1.05 <= self.minimum_hand_height_m <= 1.50:
            raise ValueError("three-role aerial-save hand gate is too low")
        if (
            min(
                self.minimum_passer_pelvis_height_m,
                self.minimum_shooter_pelvis_height_m,
                self.minimum_goalkeeper_pelvis_height_m,
            )
            < 0.55
        ):
            raise ValueError("three-role aerial-save pelvis gate is unsafe")
        if not 5.0 <= self.minimum_incoming_speed_mps <= 15.0:
            raise ValueError("three-role aerial-save incoming-speed gate is invalid")
        if not 1.0 <= self.minimum_outward_speed_mps <= 8.0:
            raise ValueError("three-role aerial-save deflection gate is invalid")
        if not 6.0 <= self.maximum_post_contact_speed_mps <= 15.0:
            raise ValueError("three-role aerial-save deflection-speed ceiling is invalid")
        if not 0.0 <= self.maximum_glove_surface_separation_m <= 0.002:
            raise ValueError("three-role aerial-save glove separation gate is invalid")
        if not 0.002 <= self.maximum_glove_surface_penetration_m <= 0.020:
            raise ValueError("three-role aerial-save glove penetration gate is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("three-role aerial-save skill is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def three_role_aerial_save_kwargs(
    *,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    config: ThreeRoleAerialSaveConfig | None = None,
) -> dict[str, Any]:
    """Compose the retained pass/shot/save policies without video authority."""

    active = config or ThreeRoleAerialSaveConfig()
    paths = (
        striker_actor_path,
        goalkeeper_actor_path,
        gmt_model_path,
        gmt_skill_path,
    )
    if any(not path.expanduser().resolve().is_file() for path in paths):
        raise ValueError("three-role aerial-save artifacts must be readable files")
    kwargs = three_role_development_kwargs()
    parent = kwargs.get("goalkeeper_config")
    if not isinstance(parent, G1GoalkeeperConfig):
        raise RuntimeError("three-role goalkeeper parent is unavailable")
    regulation_goal = G1TrainingGoalSpec(
        plane_x_m=7.50,
        width_m=7.32,
        height_m=2.44,
        depth_m=2.0,
        post_radius_m=0.06,
        target_y_m=0.89,
        target_z_m=0.115,
        precision_radius_m=0.10,
        regulation_field_enabled=True,
    )
    kwargs.update(
        goal_spec=regulation_goal,
        shooter_target=(7.50, 0.89, 0.115),
        pass_reception_target_m=active.pass_reception_target_m,
        shooter_start_sec=active.shooter_start_sec,
        shooter_ballistic_actor_path=striker_actor_path.expanduser().resolve(),
        shooter_ballistic_actor_proximity_m=0.25,
        shooter_ballistic_contact_config=G1BallisticContactResidualConfig(
            right_leg_residual_rad=(-0.0218, -0.010682792158923404, -0.06, -0.06, 0.25, 0.0305),
            contact_policy_frame=256,
            lead_duration_sec=0.1603,
            trail_duration_sec=0.065,
        ),
        shooter_ballistic_contact_torque_config=G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=(-5.0, 0.0, 0.0, 0.0, 3.75, 1.0),
            counterbalance_residual_nm=(3.6, 0.0, 0.0, 5.0, 0.0, 0.0),
            contact_policy_frame=256,
            lead_duration_sec=0.08,
            trail_duration_sec=0.065,
            maximum_joint_residual_nm=5.0,
        ),
        shooter_parameter_overrides={
            "foot_yaw_offset": active.shooter_foot_yaw_offset_rad,
            "foot_pitch_offset": active.shooter_foot_pitch_offset_rad,
            "loft_synergy": active.shooter_loft_synergy_rad,
        },
        simulation_duration_sec=active.simulation_duration_sec,
        goalkeeper_config=replace(
            parent,
            depth_from_goal_line_m=active.goalkeeper_depth_from_goal_line_m,
            actor_observation_mode="visible_ball_history_v3",
            actor_artifact_path=goalkeeper_actor_path.expanduser().resolve(),
            actor_minimum_target_height_m=1.0,
            actor_minimum_current_ball_height_m=0.35,
            actor_minimum_incoming_ball_speed_mps=0.10,
            actor_threat_warmup_sec=0.04,
            actor_minimum_intercept_confidence=0.25,
            mosaic_gmt_model_path=gmt_model_path.expanduser().resolve(),
            mosaic_gmt_skill_path=gmt_skill_path.expanduser().resolve(),
            mosaic_gmt_blend=1.0,
            mosaic_gmt_timing_lead_sec=0.0,
            mosaic_gmt_lower_body_scale=0.0,
            mosaic_gmt_waist_scale=0.35,
            mosaic_gmt_arm_scale=1.0,
            actor_bimanual_reach_enabled=True,
            actor_bimanual_reach_local_x_m=active.goalkeeper_reach_local_x_m,
            actor_bimanual_reach_half_span_m=0.06,
            actor_bimanual_reach_height_offset_m=active.goalkeeper_reach_height_offset_m,
            actor_bimanual_reach_minimum_fraction=1.0,
            actor_bimanual_reach_gain_scale=active.goalkeeper_reach_gain_scale,
            actor_bimanual_support_arm_blend=active.goalkeeper_support_arm_blend,
            actor_bimanual_support_arm_overhead_bias_rad=(
                active.goalkeeper_support_arm_overhead_bias_rad
            ),
            actor_bimanual_punch_force_n=active.goalkeeper_punch_force_n,
            actor_bimanual_punch_central_boost_n_per_m=(
                active.goalkeeper_punch_central_boost_n_per_m
            ),
            actor_bimanual_punch_reference_abs_lateral_m=(
                active.goalkeeper_punch_reference_abs_lateral_m
            ),
            glove_contact_time_constant_sec=(active.goalkeeper_glove_contact_time_constant_sec),
            glove_contact_damping_ratio=active.goalkeeper_glove_contact_damping_ratio,
            joint_guard_impact_lead_sec=0.0,
        ),
    )
    return kwargs


def evaluate_three_role_aerial_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    config: ThreeRoleAerialSaveConfig,
) -> dict[str, Any]:
    """Evaluate physical state only; rendered frames are deliberately absent."""

    if not result.finite_state:
        return {"passed": False, "reason": "non-finite rollout"}
    if result.shot_contact_time_sec is None or result.goalkeeper_glove_contact_time_sec is None:
        return {"passed": False, "reason": "missing ordered shot/glove contact"}
    time = np.asarray(trajectory["time"], dtype=np.float64)
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    shot_index = int(np.searchsorted(time, result.shot_contact_time_sec, side="left"))
    save_index = int(np.searchsorted(time, result.goalkeeper_glove_contact_time_sec, side="left"))
    if shot_index >= len(time) or save_index >= len(time) or save_index < shot_index:
        return {"passed": False, "reason": "contact timestamps exceed trajectory"}
    pre_index = max(shot_index, save_index - 1)
    post_index = min(len(time) - 1, save_index + 5)
    apex_m = float(np.max(pose[shot_index : save_index + 1, 2]))
    incoming_speed_mps = float(np.linalg.norm(velocity[pre_index, :3]))
    post_velocity = np.asarray(velocity[post_index, :3], dtype=np.float64)
    contacts_ordered = bool(
        result.pass_contact_time_sec is not None
        and result.pass_contact_time_sec
        < result.shot_contact_time_sec
        < result.goalkeeper_glove_contact_time_sec
    )
    gates = {
        "finite_state": result.finite_state,
        "contacts_ordered": contacts_ordered,
        "pass_precision": bool(
            result.pass_delivery_error_m is not None
            and result.pass_delivery_error_m <= config.minimum_pass_precision_m
            and result.pass_delivery_lateral_error_m is not None
            and result.pass_delivery_lateral_error_m <= 0.03
        ),
        "aerial_shot": bool(
            apex_m >= config.minimum_aerial_apex_m
            and incoming_speed_mps >= config.minimum_incoming_speed_mps
        ),
        "anatomical_glove_contact": bool(
            result.goalkeeper_left_glove_contact_observed
            or result.goalkeeper_right_glove_contact_observed
        ),
        "collision_faithful_glove_contact": bool(
            result.goalkeeper_glove_contact_surface_distance_m is not None
            and math.isfinite(result.goalkeeper_glove_contact_surface_distance_m)
            and -config.maximum_glove_surface_penetration_m
            <= result.goalkeeper_glove_contact_surface_distance_m
            <= config.maximum_glove_surface_separation_m
        ),
        "raised_both_hands": bool(
            result.goalkeeper_both_hands_raised_at_contact
            and result.goalkeeper_contact_left_hand_height_m is not None
            and result.goalkeeper_contact_left_hand_height_m >= config.minimum_hand_height_m
            and result.goalkeeper_contact_right_hand_height_m is not None
            and result.goalkeeper_contact_right_hand_height_m >= config.minimum_hand_height_m
        ),
        "high_glove_contact": bool(
            result.goalkeeper_glove_contact_height_m is not None
            and result.goalkeeper_glove_contact_height_m >= config.minimum_glove_contact_height_m
        ),
        "physical_save": bool(result.goalkeeper_save_observed and not result.goal_crossed),
        "outward_deflection": bool(
            max(
                -float(post_velocity[0]),
                abs(float(post_velocity[1])),
                float(post_velocity[2]),
            )
            >= config.minimum_outward_speed_mps
        ),
        "bounded_deflection_speed": bool(
            float(np.linalg.norm(post_velocity)) <= config.maximum_post_contact_speed_mps
        ),
        "three_role_stability": bool(
            result.passer_min_pelvis_height_m >= config.minimum_passer_pelvis_height_m
            and result.shooter_min_pelvis_height_m >= config.minimum_shooter_pelvis_height_m
            and result.goalkeeper_min_pelvis_height_m is not None
            and result.goalkeeper_min_pelvis_height_m >= config.minimum_goalkeeper_pelvis_height_m
        ),
        "joint_limits": not result.joint_limit_violation,
        "torque_limits": not result.torque_limit_violation,
        "zero_actuator_saturation": not result.actuator_saturation,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "aerial_apex_m": apex_m,
        "incoming_speed_mps": incoming_speed_mps,
        "post_contact_velocity_mps": tuple(float(value) for value in post_velocity),
        "trajectory_digest": trajectory_digest(trajectory),
        "result": result.to_dict(),
    }


def run_three_role_aerial_save_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: ThreeRoleAerialSaveConfig | None = None,
) -> dict[str, Any]:
    """Run two strict CPU replays and freeze only a fully passing trajectory."""

    active = config or ThreeRoleAerialSaveConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("three-role aerial-save evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    kwargs = three_role_aerial_save_kwargs(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        config=active,
    )
    artifacts = {
        "striker_actor_hash": hash_bytes(striker_actor_path.read_bytes()),
        "goalkeeper_actor_hash": hash_bytes(goalkeeper_actor_path.read_bytes()),
        "gmt_model_hash": hash_bytes(gmt_model_path.read_bytes()),
        "gmt_skill_hash": hash_bytes(gmt_skill_path.read_bytes()),
    }
    request = {
        "schema_version": "rosclaw_soccer.three_role_aerial_save_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "goal_spec": asdict(kwargs["goal_spec"]),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "artifacts": artifacts,
        "source_commit": _git_head(checkout),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    output.mkdir(parents=True)
    _write_json(output / "request.json", request)
    first_result, first_trajectory = simulate_shared_world(asset_root, **kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
    first = evaluate_three_role_aerial_save(
        result=first_result,
        trajectory=first_trajectory,
        config=active,
    )
    replay = evaluate_three_role_aerial_save(
        result=replay_result,
        trajectory=replay_trajectory,
        config=active,
    )
    strict_replay = bool(
        first_result.to_dict() == replay_result.to_dict()
        and first["trajectory_digest"] == replay["trajectory_digest"]
    )
    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
    report = {
        "schema_version": "rosclaw_soccer.three_role_aerial_save_evidence.v1",
        "passed": bool(first.get("passed") and replay.get("passed") and strict_replay),
        "promotion_status": "FROZEN_SIM_DEMO"
        if first.get("passed") and replay.get("passed") and strict_replay
        else "REJECTED_DEVELOPMENT",
        "strict_replay": strict_replay,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
        "simultaneous_three_body_physics": True,
        "single_shared_ball": True,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
        "first": first,
        "replay": replay,
    }
    _write_json(output / "evidence.json", report)
    return report


def _implementation_hash() -> str:
    shared = Path(__file__).parents[1] / "skills" / "team" / "shared_world.py"
    return str(
        hash_json(
            {
                "evidence_loop": hash_bytes(Path(__file__).read_bytes()),
                "shared_world": hash_bytes(shared.read_bytes()),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ThreeRoleAerialSaveConfig",
    "evaluate_three_role_aerial_save",
    "run_three_role_aerial_save_evidence",
    "three_role_aerial_save_kwargs",
]
