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
from typing import Any, TypeVar, cast

from rosclaw_soccer.physics.standards import IFABRegulationSpec

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
    football_motion_prior: Path


@dataclass(frozen=True)
class Age04RegulationCurriculum:
    goal_plane_x_m: float = 8.5
    target_y_m: float = 1.32
    target_z_m: float = 1.04
    precision_radius_m: float = 0.10
    net_capture_depth_m: float = 1.35
    teacher_force_pairs_n: tuple[tuple[float, float], ...] = (
        (250.0, 250.0),
        (245.0, 245.0),
        (250.0, 10.0),
        (10.0, 10.0),
        (20.0, 10.0),
        (30.0, 10.0),
        (10.0, 20.0),
        (10.0, 30.0),
    )
    schema_version: str = "rosclaw_soccer.age04_regulation_curriculum.v1"

    def __post_init__(self) -> None:
        values = (
            self.goal_plane_x_m,
            self.target_y_m,
            self.target_z_m,
            self.precision_radius_m,
            self.net_capture_depth_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Age-4 curriculum values must be finite")
        if len(self.teacher_force_pairs_n) != 8:
            raise ValueError("Age-4 actor curriculum requires exactly eight teacher probes")
        if len(set(self.teacher_force_pairs_n)) != len(self.teacher_force_pairs_n):
            raise ValueError("Age-4 teacher probes must be unique")
        if not all(
            10.0 <= lateral <= 250.0 and 10.0 <= vertical <= 250.0
            for lateral, vertical in self.teacher_force_pairs_n
        ):
            raise ValueError("Age-4 teacher forces must be in [10, 250] N")


@dataclass(frozen=True)
class Age04RegulationTrainingReport:
    passed: bool
    actor_hash: str
    final_evidence_path: str
    final_goal_plane_target_error_m: float | None
    final_goal_crossing_xyz_m: tuple[float, float, float] | None
    final_ball_retained_in_goal: bool
    final_post_kick_fall: bool
    final_post_contact_backward_displacement_m: float
    probe_evidence_paths: tuple[str, ...]
    output_path: str
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.age04_regulation_training_report.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        aim_bias_z_m=0.20,
        shot_loft_synergy_rad=0.30,
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
    motion_prior = load_g1_football_motion_prior(assets.football_motion_prior)
    residual = G1ApproachStrikeResidualConfig(
        residual_fraction=0.2,
        maximum_residual_nm=5.0,
    )
    shared: dict[str, Any] = {
        "asset_root": assets.asset_root,
        "gait_policy_root": assets.gait_policy_root,
        "source_checkout": checkout,
        "runup_config": runup,
        "goal_spec": goal,
        "sonic_model_root": assets.sonic_model_root,
        "sonic_runup_config": sonic,
        "approach_strike_candidate_path": assets.approach_strike_candidate,
        "approach_strike_residual_config": residual,
        "football_motion_prior": motion_prior,
    }

    evidence_paths: list[Path] = []
    for index, (lateral_force, vertical_force) in enumerate(academy.teacher_force_pairs_n):
        probe_flow = replace(
            flow,
            shot_loft_teacher_target_vy_mps=10.0,
            shot_loft_teacher_target_vz_mps=7.0,
            shot_loft_teacher_lateral_gain_n_per_mps=35.0,
            shot_loft_teacher_gain_n_per_mps=50.0,
            shot_loft_teacher_max_lateral_force_n=lateral_force,
            shot_loft_teacher_max_force_n=vertical_force,
            shot_loft_teacher_max_foot_ball_distance_m=0.18,
        )
        probe_dir = root / f"probe-{index:02d}-l{lateral_force:.0f}-v{vertical_force:.0f}"
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
    report_path = root / "age04-regulation-training.json"
    report = Age04RegulationTrainingReport(
        passed=final.passed,
        actor_hash=actor.actor_hash,
        final_evidence_path=str(final_dir / "g1-free-kick.json"),
        final_goal_plane_target_error_m=result.goal_plane_target_error_m,
        final_goal_crossing_xyz_m=result.goal_crossing_xyz_m,
        final_ball_retained_in_goal=result.ball_retained_in_goal,
        final_post_kick_fall=result.post_kick_fall,
        final_post_contact_backward_displacement_m=result.post_contact_backward_displacement_m,
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


__all__ = [
    "Age04RegulationAssets",
    "Age04RegulationCurriculum",
    "Age04RegulationTrainingReport",
    "run_age04_regulation_training",
]
