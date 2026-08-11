"""Train an Age-4 contact actor against regulation football physics.

This curriculum deliberately derives a fresh actor instead of relabelling the
legacy small-goal actor. Eight teacher probes share one frozen context: two
high-authority precision candidates and six bounded negative controls. The
derived actor is then replayed without the teacher in one continuous MuJoCo
world. All artifacts remain SIM_ONLY and outside the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from rosclaw_soccer.physics.standards import IFABRegulationSpec

if TYPE_CHECKING:
    from rosclaw.simforge.g1_free_kick_showcase import G1FreeKickEvidence

_ConfigT = TypeVar("_ConfigT")
_TUPLE_FIELDS = {
    "joint_gain_scales",
    "strike_gain_scales",
    "follow_through_gain_scales",
    "ballistic_contact_residual_rad",
    "ballistic_contact_torque_residual_nm",
    "ballistic_contact_torque_preload_nm",
    "ballistic_contact_torque_phase_offset_sec",
    "ballistic_counterbalance_torque_residual_nm",
}


@dataclass(frozen=True)
class Age04RegulationAssets:
    asset_root: Path
    gait_policy_root: Path
    sonic_model_root: Path
    seed_request: Path
    approach_strike_candidate: Path
    football_motion_prior: Path | None = None


@dataclass(frozen=True)
class Age04RegulationCurriculum:
    goal_plane_x_m: float = 8.5
    target_y_m: float = 1.32
    target_z_m: float = 1.04
    precision_radius_m: float = 0.10
    net_capture_depth_m: float = 1.35
    aim_bias_y_m: float = 1.12
    aim_bias_z_m: float = 0.205
    shot_loft_synergy_rad: float = 0.30
    shot_foot_yaw_offset_rad: float = 0.02
    ballistic_contact_policy_frame: int = 258
    ballistic_contact_torque_policy_frame: int = 258
    ballistic_contact_residual_rad: tuple[float, ...] = (
        -0.0218,
        -0.010682792158923404,
        -0.06,
        -0.06,
        0.225,
        0.0305,
    )
    sonic_planner_seed: int = 21
    sonic_execution_duration_sec: float = 3.95
    residual_fraction: float = 0.50
    maximum_residual_nm: float = 20.0
    maximum_standardized_rms: float = 20.0
    maximum_standardized_abs: float = 100.0
    residual_active_event_phase_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    support_chain_event_phase_id: int = 4
    support_chain_left_knee_delta_nm: float = -1.20
    football_motion_prior_blend: float = 0.0
    torque_authority_projection_ratio: float = 0.98
    torque_authority_projection_max_fraction: float = 0.01
    teacher_velocity_gain_n_per_mps: float = 50.0
    teacher_probe_specs: tuple[tuple[float, float, float], ...] = (
        (7.0, 80.0, 50.0),
        (7.0, 80.0, 55.0),
        (7.0, 80.0, 60.0),
        (7.0, 80.0, 65.0),
        (7.0, 80.0, 70.0),
        (-1.0, 80.0, 10.0),
        (-2.0, 80.0, 20.0),
        (-3.0, 80.0, 30.0),
    )
    schema_version: str = "rosclaw_soccer.age04_regulation_curriculum.v2"

    def __post_init__(self) -> None:
        values = (
            self.goal_plane_x_m,
            self.target_y_m,
            self.target_z_m,
            self.precision_radius_m,
            self.net_capture_depth_m,
            self.aim_bias_y_m,
            self.aim_bias_z_m,
            self.shot_loft_synergy_rad,
            self.shot_foot_yaw_offset_rad,
            self.sonic_execution_duration_sec,
            self.residual_fraction,
            self.maximum_residual_nm,
            self.maximum_standardized_rms,
            self.maximum_standardized_abs,
            self.support_chain_left_knee_delta_nm,
            self.football_motion_prior_blend,
            self.torque_authority_projection_ratio,
            self.torque_authority_projection_max_fraction,
            self.teacher_velocity_gain_n_per_mps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Age-4 curriculum values must be finite")
        if len(self.teacher_probe_specs) != 8:
            raise ValueError("Age-4 actor curriculum requires exactly eight teacher probes")
        if len(set(self.teacher_probe_specs)) != len(self.teacher_probe_specs):
            raise ValueError("Age-4 teacher probes must be unique")
        if not all(
            (-4.0 <= target_vertical <= -0.5 or 3.0 <= target_vertical <= 7.0)
            and 10.0 <= lateral <= 250.0
            and 10.0 <= vertical <= 250.0
            for target_vertical, lateral, vertical in self.teacher_probe_specs
        ):
            raise ValueError("Age-4 teacher probe speeds or forces are outside bounds")
        if not 0.5 <= self.aim_bias_y_m <= 1.5 or not 0.0 <= self.aim_bias_z_m <= 0.5:
            raise ValueError("Age-4 aim bias is outside the bounded curriculum")
        if not 0.0 <= self.shot_loft_synergy_rad <= 0.30:
            raise ValueError("Age-4 loft synergy must be in [0, 0.30] rad")
        if not -0.10 <= self.shot_foot_yaw_offset_rad <= 0.10:
            raise ValueError("Age-4 foot yaw offset must be in [-0.10, 0.10] rad")
        if len(self.ballistic_contact_residual_rad) != 6 or not all(
            math.isfinite(value) and abs(value) <= 0.25
            for value in self.ballistic_contact_residual_rad
        ):
            raise ValueError("Age-4 contact residual must contain six bounded values")
        if not 150 <= self.ballistic_contact_policy_frame <= 430 or not (
            150 <= self.ballistic_contact_torque_policy_frame <= 430
        ):
            raise ValueError("Age-4 contact policy frames must be in [150, 430]")
        if not 0 <= self.sonic_planner_seed <= 1_000_000:
            raise ValueError("Age-4 SONIC planner seed must be in [0, 1000000]")
        if not 3.0 <= self.sonic_execution_duration_sec <= 4.5:
            raise ValueError("Age-4 SONIC duration must be in [3.0, 4.5] s")
        if not 0.0 < self.residual_fraction <= 0.50:
            raise ValueError("Age-4 residual fraction must be in (0, 0.50]")
        if not 0.10 <= self.maximum_residual_nm <= 20.0:
            raise ValueError("Age-4 maximum residual must be in [0.10, 20] Nm")
        if not 0.50 <= self.maximum_standardized_rms <= 20.0:
            raise ValueError("Age-4 standardized RMS bound must be in [0.50, 20]")
        if not 1.0 <= self.maximum_standardized_abs <= 100.0:
            raise ValueError("Age-4 standardized absolute bound must be in [1, 100]")
        if (
            not self.residual_active_event_phase_ids
            or len(set(self.residual_active_event_phase_ids))
            != len(self.residual_active_event_phase_ids)
            or not set(self.residual_active_event_phase_ids).issubset({0, 1, 2, 3, 4})
        ):
            raise ValueError("Age-4 residual active phases must be unique ids in [0, 4]")
        if self.support_chain_event_phase_id not in self.residual_active_event_phase_ids:
            raise ValueError("Age-4 support-chain phase must be active")
        if not -20.0 <= self.support_chain_left_knee_delta_nm <= -0.10:
            raise ValueError("Age-4 support-chain knee delta must be in [-20, -0.10] Nm")
        if not 0.0 <= self.football_motion_prior_blend <= 0.35:
            raise ValueError("Age-4 motion-prior blend must be in [0, 0.35]")
        if not 5.0 <= self.teacher_velocity_gain_n_per_mps <= 50.0:
            raise ValueError("Age-4 teacher velocity gain must be in [5, 50]")
        if not 0.90 <= self.torque_authority_projection_ratio <= 0.99:
            raise ValueError("Age-4 torque authority ratio must be in [0.90, 0.99]")
        if not 0.001 <= self.torque_authority_projection_max_fraction <= 0.05:
            raise ValueError("Age-4 torque authority fraction must be in [0.001, 0.05]")


@dataclass(frozen=True)
class Age04RegulationTrainingReport:
    passed: bool
    verdict: str
    failure_codes: tuple[str, ...]
    provenance_passed: bool
    precision_passed: bool
    continuity_passed: bool
    dynamic_stability_passed: bool
    recovery_passed: bool
    torque_authority_passed: bool
    development_breakthrough: bool
    core_showcase_passed: bool
    support_candidate_hash: str
    support_candidate_path: str
    actor_hash: str
    final_evidence_path: str
    final_goal_plane_target_error_m: float | None
    final_goal_crossing_xyz_m: tuple[float, float, float] | None
    final_ball_retained_in_goal: bool
    final_post_kick_fall: bool
    final_post_contact_backward_displacement_m: float
    final_actuator_saturation_steps: int
    final_torque_authority_projection_steps: int
    final_torque_authority_projection_fraction: float
    final_torque_authority_projection_peak_correction_nm: float
    final_torque_authority_preprojection_peak_demand_ratio: float
    final_torque_authority_projection_qualified: bool
    final_contact_task_authority_projection_steps: int
    final_contact_task_authority_scale_min: float
    final_joint_boundary_guard_active_steps: int
    final_joint_boundary_guard_peak_correction_nm: float
    final_perceptual_continuity_passed: bool
    final_runup_min_pelvis_height_m: float
    final_runup_peak_tilt_rad: float
    final_runup_terminal_speed_mps: float
    final_kick_min_pelvis_height_m: float
    final_kick_peak_tilt_rad: float
    final_post_contact_settling_time_sec: float
    final_post_contact_final_joint_velocity_rms_rad_s: float
    probe_evidence_paths: tuple[str, ...]
    output_path: str
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.age04_regulation_training_report.v3"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Age04RegulationAssessment:
    """Independent, multi-axis gate for the regulation-target curriculum.

    The core free-kick showcase intentionally certifies a declared goal corner
    and a narrow net-capture location.  Age 4 currently trains a regulation
    placement target inside the goal, so reusing that aggregate Boolean would
    conflate contact learning with a different exam.  This gate preserves the
    same physical safety and stability requirements while scoring precision at
    the curriculum target.  It never turns a stability failure into a pass.
    """

    passed: bool
    verdict: str
    failure_codes: tuple[str, ...]
    provenance_passed: bool
    precision_passed: bool
    continuity_passed: bool
    dynamic_stability_passed: bool
    recovery_passed: bool
    torque_authority_passed: bool
    development_breakthrough: bool
    schema_version: str = "rosclaw_soccer.age04_regulation_assessment.v1"


def assess_age04_regulation(
    evidence: G1FreeKickEvidence,
    curriculum: Age04RegulationCurriculum,
) -> Age04RegulationAssessment:
    """Score one teacher-free replay without inheriting an unrelated corner gate."""

    result = evidence.result
    sonic_eligible = evidence.sonic_qualification is not None and bool(
        evidence.sonic_qualification.eligible
    )
    provenance_passed = bool(
        evidence.strict_replay
        and evidence.learned_gait_qualification.eligible
        and sonic_eligible
        and evidence.activation_ceiling == "SIM_ONLY"
        and not evidence.hardware_command_sent
    )
    precision_passed = bool(
        result.finite_state
        and result.learned_runup_executed
        and result.ballistic_contact_impulse_actor_executed
        and not result.loft_teacher_executed
        and result.continuous_single_world
        and not result.state_reset_after_start
        and result.kick_contact_observed
        and result.goal_crossed
        and result.goal_mouth_hit
        and _finite_at_most(result.goal_plane_target_error_m, curriculum.precision_radius_m)
        and result.ball_retained_in_goal
    )
    continuity_passed = bool(
        result.perceptual_continuity_passed
        and _finite_at_least(result.initial_ball_distance_m, 4.0)
        and _finite_at_least(result.runup_distance_m, 3.0)
        and _finite_at_least(result.runup_peak_speed_mps, 1.0)
    )
    dynamic_stability_passed = bool(
        _finite_at_least(result.runup_min_pelvis_height_m, 0.70)
        and _finite_at_most(result.runup_peak_tilt_rad, 0.30)
        and _finite_between(result.runup_terminal_speed_mps, 0.10, 0.40)
        and _finite_at_least(result.kick_min_pelvis_height_m, 0.68)
        and _finite_at_most(result.kick_peak_tilt_rad, 0.40)
    )
    recovery_passed = bool(
        not result.post_kick_fall
        and not result.joint_limit_violation
        and _finite_at_least(result.final_pelvis_height_m, 0.70)
        and _finite_at_most(result.final_speed_mps, 0.20)
        and _finite_at_most(result.post_contact_backward_displacement_m, 0.20)
        and result.post_contact_forward_velocity_reversals <= 12
        and _finite_at_most(result.post_contact_settling_time_sec, 5.0)
        and _finite_at_most(result.post_contact_final_joint_velocity_rms_rad_s, 0.10)
        and _finite_at_most(result.post_contact_mean_pelvis_speed_mps, 0.12)
        and _finite_at_most(result.post_contact_mean_joint_velocity_rms_rad_s, 0.25)
    )
    torque_authority_passed = bool(
        not result.torque_limit_violation
        and not result.actuator_saturation
        and result.actuator_saturation_steps == 0
        and result.torque_authority_projection_qualified
        and _finite_at_most(
            result.torque_authority_projection_fraction,
            curriculum.torque_authority_projection_max_fraction,
        )
        and _finite_at_least(result.contact_task_authority_scale_min, 0.95)
    )
    axes = {
        "PROVENANCE_GATE": provenance_passed,
        "PRECISION_GATE": precision_passed,
        "CONTINUITY_GATE": continuity_passed,
        "DYNAMIC_STABILITY_GATE": dynamic_stability_passed,
        "RECOVERY_GATE": recovery_passed,
        "TORQUE_AUTHORITY_GATE": torque_authority_passed,
    }
    failure_codes = tuple(name for name, passed in axes.items() if not passed)
    passed = not failure_codes
    development_breakthrough = bool(
        provenance_passed
        and precision_passed
        and continuity_passed
        and recovery_passed
        and torque_authority_passed
    )
    verdict = "PASS" if passed else "DEVELOPMENT" if development_breakthrough else "REJECTED"
    return Age04RegulationAssessment(
        passed=passed,
        verdict=verdict,
        failure_codes=failure_codes,
        provenance_passed=provenance_passed,
        precision_passed=precision_passed,
        continuity_passed=continuity_passed,
        dynamic_stability_passed=dynamic_stability_passed,
        recovery_passed=recovery_passed,
        torque_authority_passed=torque_authority_passed,
        development_breakthrough=development_breakthrough,
    )


def run_age04_regulation_training(
    *,
    assets: Age04RegulationAssets,
    output_dir: Path,
    source_checkout: Path,
    curriculum: Age04RegulationCurriculum | None = None,
    standard: IFABRegulationSpec | None = None,
) -> Age04RegulationTrainingReport:
    """Run eight bounded probes, distil an actor, and replay it teacher-free."""

    from rosclaw.growth.approach_strike_residual import G1ApproachStrikeResidualConfig
    from rosclaw.growth.ballistic_contact_impulse_actor import (
        derive_g1_ballistic_contact_impulse_actor,
    )
    from rosclaw.growth.football_motion_prior import load_g1_football_motion_prior
    from rosclaw.growth.phase_conditioned_residual import (
        G1PhaseConditionedResidualConfig,
        derive_g1_phase_conditioned_residual_candidate,
    )
    from rosclaw.simforge.g1_free_kick_showcase import (
        G1FreeKickFlowConfig,
        run_g1_free_kick_showcase,
    )
    from rosclaw.simforge.g1_learned_runup import G1LearnedRunupConfig
    from rosclaw.simforge.g1_sonic_runup import G1SonicRunupConfig
    from rosclaw.simforge.g1_stadium_scene import G1TrainingGoalSpec

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Age-4 training evidence must remain outside the source checkout")
    if root.exists():
        raise FileExistsError("Age-4 training output already exists")
    root.mkdir(parents=True)

    academy = curriculum or Age04RegulationCurriculum()
    regulation = standard or IFABRegulationSpec()
    seed = _read_json(assets.seed_request)
    runup = _config_from_json(G1LearnedRunupConfig, dict(seed["runup_config"]))
    sonic = _config_from_json(G1SonicRunupConfig, dict(seed["sonic_runup_config"]))
    sonic = replace(
        sonic,
        planner_seed=academy.sonic_planner_seed,
        execution_duration_sec=academy.sonic_execution_duration_sec,
    )
    flow = _config_from_json(G1FreeKickFlowConfig, dict(seed["flow_config"]))
    flow = replace(
        flow,
        ballistic_contact_impulse_actor_hash=None,
        net_capture_depth_m=academy.net_capture_depth_m,
        net_stiffness_n_m=65.0,
        net_damping_n_s_m=12.0,
        shared_cerebellar_recovery_enabled=True,
        shot_recovery_step_length_m=0.11,
        shot_recovery_step_yaw_rad=-0.05,
        post_contact_damping_delay_sec=0.18,
        post_contact_damping_ramp_sec=0.45,
        torque_authority_projection_ratio=academy.torque_authority_projection_ratio,
        torque_authority_projection_max_fraction=(academy.torque_authority_projection_max_fraction),
        aim_bias_y_m=academy.aim_bias_y_m,
        aim_bias_z_m=academy.aim_bias_z_m,
        shot_loft_synergy_rad=academy.shot_loft_synergy_rad,
        shot_foot_yaw_offset_rad=academy.shot_foot_yaw_offset_rad,
        ballistic_contact_policy_frame=academy.ballistic_contact_policy_frame,
        ballistic_contact_torque_policy_frame=(academy.ballistic_contact_torque_policy_frame),
        ballistic_contact_residual_rad=academy.ballistic_contact_residual_rad,
        football_motion_prior_hash=None,
        football_motion_prior_blend=academy.football_motion_prior_blend,
        shot_loft_teacher_gain_n_per_mps=academy.teacher_velocity_gain_n_per_mps,
    )
    goal = G1TrainingGoalSpec(
        plane_x_m=academy.goal_plane_x_m,
        width_m=regulation.goal_inside_width_m,
        height_m=regulation.goal_inside_height_m,
        depth_m=regulation.net_depth_m,
        post_radius_m=regulation.goal_frame_radius_m,
        net_strand_radius_m=0.003,
        target_y_m=academy.target_y_m,
        target_z_m=academy.target_z_m,
        precision_radius_m=academy.precision_radius_m,
        ball_free_joint_damping_n_s_m=regulation.ball_linear_damping_n_s_m,
        ball_radius_m=regulation.ball_radius_m,
        ball_mass_kg=regulation.ball_mass_kg,
        ball_contact_sliding_friction=0.05,
        ball_sliding_friction=regulation.ball_sliding_friction,
        ball_torsional_friction=regulation.ball_torsional_friction,
        ball_rolling_friction=regulation.ball_rolling_friction,
        regulation_field_enabled=True,
        field_length_m=regulation.field_length_m,
        field_width_m=regulation.field_width_m,
        field_line_width_m=regulation.line_width_m,
        goal_area_depth_m=regulation.goal_area_depth_m,
        penalty_area_depth_m=regulation.penalty_area_depth_m,
        penalty_mark_distance_m=regulation.penalty_mark_distance_m,
    )
    motion_prior = None
    if academy.football_motion_prior_blend > 0.0:
        if assets.football_motion_prior is None:
            raise ValueError("Age-4 motion-prior blend requires a motion-prior artifact")
        motion_prior = load_g1_football_motion_prior(assets.football_motion_prior)
        flow = replace(flow, football_motion_prior_hash=motion_prior.prior_hash)
    support_delta = [0.0] * 29
    support_delta[3] = academy.support_chain_left_knee_delta_nm
    support_candidate_path = derive_g1_phase_conditioned_residual_candidate(
        base_candidate_path=assets.approach_strike_candidate,
        output_dir=root / "support-chain-candidate",
        source_checkout=checkout,
        config=G1PhaseConditionedResidualConfig(
            event_phase_id=academy.support_chain_event_phase_id,
            joint_delta_nm=tuple(support_delta),
        ),
    )
    support_candidate = _read_json(support_candidate_path)
    support_candidate_hash = str(support_candidate["candidate_hash"])
    residual = G1ApproachStrikeResidualConfig(
        residual_fraction=academy.residual_fraction,
        maximum_residual_nm=academy.maximum_residual_nm,
        maximum_standardized_rms=academy.maximum_standardized_rms,
        maximum_standardized_abs=academy.maximum_standardized_abs,
        active_event_phase_ids=academy.residual_active_event_phase_ids,
    )
    shared: dict[str, Any] = {
        "asset_root": assets.asset_root,
        "gait_policy_root": assets.gait_policy_root,
        "source_checkout": checkout,
        "runup_config": runup,
        "goal_spec": goal,
        "sonic_model_root": assets.sonic_model_root,
        "sonic_runup_config": sonic,
        "approach_strike_candidate_path": support_candidate_path,
        "approach_strike_residual_config": residual,
    }
    if motion_prior is not None:
        shared["football_motion_prior"] = motion_prior

    evidence_paths: list[Path] = []
    for index, (target_vertical, lateral_force, vertical_force) in enumerate(
        academy.teacher_probe_specs
    ):
        probe_flow = replace(
            flow,
            shot_loft_teacher_target_vy_mps=0.0,
            shot_loft_teacher_target_vz_mps=target_vertical,
            shot_loft_teacher_gain_n_per_mps=(academy.teacher_velocity_gain_n_per_mps),
            shot_loft_teacher_max_lateral_force_n=lateral_force,
            shot_loft_teacher_max_force_n=vertical_force,
            shot_loft_teacher_max_foot_ball_distance_m=0.18,
        )
        probe_dir = root / (
            f"probe-{index:02d}-vz{target_vertical:+.0f}-l{lateral_force:.0f}-v{vertical_force:.0f}"
        )
        run_g1_free_kick_showcase(output_dir=probe_dir, flow_config=probe_flow, **shared)
        evidence_paths.append(probe_dir / "g1-free-kick.json")

    actor_path = root / "age04-regulation-contact-actor.json"
    actor = derive_g1_ballistic_contact_impulse_actor(
        evidence_paths=tuple(evidence_paths),
        output_path=actor_path,
        source_checkout=checkout,
    )
    final_flow = replace(
        flow,
        ballistic_contact_impulse_actor_hash=actor.actor_hash,
        shot_loft_teacher_target_vy_mps=0.0,
        shot_loft_teacher_target_vz_mps=0.0,
        shot_loft_teacher_max_foot_ball_distance_m=0.0,
    )
    final_dir = root / "final-teacher-free"
    final = run_g1_free_kick_showcase(
        output_dir=final_dir,
        flow_config=final_flow,
        ballistic_contact_impulse_actor=actor,
        **shared,
    )
    result = final.result
    assessment = assess_age04_regulation(final, academy)
    report_path = root / "age04-regulation-training.json"
    report = Age04RegulationTrainingReport(
        passed=assessment.passed,
        verdict=assessment.verdict,
        failure_codes=assessment.failure_codes,
        provenance_passed=assessment.provenance_passed,
        precision_passed=assessment.precision_passed,
        continuity_passed=assessment.continuity_passed,
        dynamic_stability_passed=assessment.dynamic_stability_passed,
        recovery_passed=assessment.recovery_passed,
        torque_authority_passed=assessment.torque_authority_passed,
        development_breakthrough=assessment.development_breakthrough,
        core_showcase_passed=final.passed,
        support_candidate_hash=support_candidate_hash,
        support_candidate_path=str(support_candidate_path),
        actor_hash=actor.actor_hash,
        final_evidence_path=str(final_dir / "g1-free-kick.json"),
        final_goal_plane_target_error_m=result.goal_plane_target_error_m,
        final_goal_crossing_xyz_m=result.goal_crossing_xyz_m,
        final_ball_retained_in_goal=result.ball_retained_in_goal,
        final_post_kick_fall=result.post_kick_fall,
        final_post_contact_backward_displacement_m=result.post_contact_backward_displacement_m,
        final_actuator_saturation_steps=result.actuator_saturation_steps,
        final_torque_authority_projection_steps=result.torque_authority_projection_steps,
        final_torque_authority_projection_fraction=(result.torque_authority_projection_fraction),
        final_torque_authority_projection_peak_correction_nm=(
            result.torque_authority_projection_peak_correction_nm
        ),
        final_torque_authority_preprojection_peak_demand_ratio=(
            result.torque_authority_preprojection_peak_demand_ratio
        ),
        final_torque_authority_projection_qualified=(result.torque_authority_projection_qualified),
        final_contact_task_authority_projection_steps=(
            result.contact_task_authority_projection_steps
        ),
        final_contact_task_authority_scale_min=result.contact_task_authority_scale_min,
        final_joint_boundary_guard_active_steps=result.joint_boundary_guard_active_steps,
        final_joint_boundary_guard_peak_correction_nm=(
            result.joint_boundary_guard_peak_correction_nm
        ),
        final_perceptual_continuity_passed=result.perceptual_continuity_passed,
        final_runup_min_pelvis_height_m=result.runup_min_pelvis_height_m,
        final_runup_peak_tilt_rad=result.runup_peak_tilt_rad,
        final_runup_terminal_speed_mps=result.runup_terminal_speed_mps,
        final_kick_min_pelvis_height_m=result.kick_min_pelvis_height_m,
        final_kick_peak_tilt_rad=result.kick_peak_tilt_rad,
        final_post_contact_settling_time_sec=result.post_contact_settling_time_sec,
        final_post_contact_final_joint_velocity_rms_rad_s=(
            result.post_contact_final_joint_velocity_rms_rad_s
        ),
        probe_evidence_paths=tuple(str(path) for path in evidence_paths),
        output_path=str(report_path),
    )
    unsigned = report.to_dict()
    unsigned["curriculum"] = asdict(academy)
    unsigned["standard"] = regulation.to_dict()
    unsigned["actor_artifact_sha256"] = _hash_file(actor_path)
    unsigned["report_hash"] = _hash_json(unsigned)
    report_path.write_text(json.dumps(unsigned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _config_from_json(cls: type[_ConfigT], payload: dict[str, Any]) -> _ConfigT:
    accepted = set(cast(dict[str, Any], cast(Any, cls).__dataclass_fields__))
    values = {
        key: tuple(value) if key in _TUPLE_FIELDS and isinstance(value, list) else value
        for key, value in payload.items()
        if key in accepted and key != "schema_version"
    }
    return cls(**values)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Age-4 seed request must be a JSON object")
    return cast(dict[str, Any], value)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _finite_at_most(value: float | None, maximum: float) -> bool:
    return value is not None and math.isfinite(value) and value <= maximum


def _finite_at_least(value: float | None, minimum: float) -> bool:
    return value is not None and math.isfinite(value) and value >= minimum


def _finite_between(value: float | None, minimum: float, maximum: float) -> bool:
    return _finite_at_least(value, minimum) and _finite_at_most(value, maximum)


__all__ = [
    "Age04RegulationAssets",
    "Age04RegulationAssessment",
    "Age04RegulationCurriculum",
    "Age04RegulationTrainingReport",
    "assess_age04_regulation",
    "run_age04_regulation_training",
]
