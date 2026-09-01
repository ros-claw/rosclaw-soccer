"""Physics-grounded moving-ball First Touch acquisition for a qualified G1.

The runner reuses the Soccer shared MuJoCo world and frozen RoboNaldo whole-
body prior.  It may discover bounded parameter adapters, but it cannot command
hardware or promote a policy.  Rendered pixels never determine a metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.first_touch import (
    FirstTouchEvaluation,
    FirstTouchGateConfig,
    FirstTouchMeasurement,
    evaluate_first_touch,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    G1AssetQualification,
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import ShotParameters, hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1JointGuardConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class FirstTouchPhysicsScenario:
    scenario_id: str
    incoming_speed_mps: float
    incoming_lateral_m: float = 0.0
    target_direction_deg: float = 0.0
    target_outgoing_speed_mps: float = 1.2
    measurement_horizon_sec: float = 0.20
    simulation_duration_sec: float = 8.0
    frozen_partner_origin_m: tuple[float, float, float] = (3.745, -3.0, 0.0)
    frozen_policy_target_m: tuple[float, float, float] = (5.0, 0.8, 1.0)
    seed: int = 118
    schema_version: str = "rosclaw_soccer.first_touch_physics_scenario.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.scenario_id):
            raise ValueError("First Touch scenario identity is invalid")
        values = (
            self.incoming_speed_mps,
            self.incoming_lateral_m,
            self.target_direction_deg,
            self.target_outgoing_speed_mps,
            self.measurement_horizon_sec,
            self.simulation_duration_sec,
            *self.frozen_partner_origin_m,
            *self.frozen_policy_target_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("First Touch scenario values must be finite")
        if not 0.5 <= self.incoming_speed_mps <= 2.0:
            raise ValueError("First Touch acquisition speed must be in [0.5, 2.0] m/s")
        if not -0.20 <= self.incoming_lateral_m <= 0.20:
            raise ValueError("First Touch lateral offset exceeds the acquisition pocket")
        if not -45.0 <= self.target_direction_deg <= 45.0:
            raise ValueError("First Touch target direction exceeds the acquisition cone")
        if not 0.15 <= self.target_outgoing_speed_mps <= 2.5:
            raise ValueError("First Touch target speed is outside the controlled-touch gate")
        if not 0.10 <= self.measurement_horizon_sec <= 0.40:
            raise ValueError("First Touch measurement horizon is invalid")
        if not 7.0 <= self.simulation_duration_sec <= 12.0:
            raise ValueError("First Touch simulation duration is invalid")
        if (
            len(self.frozen_partner_origin_m) != 3
            or abs(self.frozen_partner_origin_m[1]) < 2.0
            or self.frozen_partner_origin_m[2] != 0.0
        ):
            raise ValueError("frozen partner must remain outside the First Touch lane")
        if (
            len(self.frozen_policy_target_m) != 3
            or not 4.0 <= self.frozen_policy_target_m[0] <= 6.0
            or not -1.0 <= self.frozen_policy_target_m[1] <= 1.0
            or not 0.115 <= self.frozen_policy_target_m[2] <= 1.4
        ):
            raise ValueError("frozen policy target exceeds its qualified coordinate frame")
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("First Touch scenario seed is invalid")

    @property
    def scenario_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def launcher_position_m(self) -> tuple[float, float, float]:
        # The frozen policy's contact phase is near six seconds in this world.
        # The coefficient is intentionally conservative: physical phase-sync,
        # not this estimate, owns the eventual contact timing.
        return (
            1.10 + 3.80 * self.incoming_speed_mps,
            self.incoming_lateral_m,
            0.115,
        )

    @property
    def launcher_velocity_mps(self) -> tuple[float, float, float]:
        return (-self.incoming_speed_mps, 0.0, 0.0)


@dataclass(frozen=True)
class FirstTouchCandidate:
    candidate_id: str
    kick_foot: str = "right"
    receiver_start_delay_sec: float = 0.0
    stance_offset_x: float = 0.0
    stance_offset_y: float = -0.06
    swing_amplitude: float = 1.0
    swing_speed_scale: float = 0.90
    com_shift_y: float = -0.065
    pelvis_yaw_offset: float = 0.175
    foot_yaw_offset: float = 0.03025
    foot_pitch_offset: float = 0.0
    loft_synergy: float = 0.0
    schema_version: str = "rosclaw_soccer.first_touch_candidate.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.candidate_id):
            raise ValueError("First Touch candidate identity is invalid")
        if (
            not math.isfinite(self.receiver_start_delay_sec)
            or not 0.0 <= self.receiver_start_delay_sec <= 1.0
        ):
            raise ValueError("First Touch receiver start delay must be in [0, 1] seconds")
        ShotParameters(
            stance_offset_x=self.stance_offset_x,
            stance_offset_y=self.stance_offset_y,
            kick_foot=self.kick_foot,
            swing_amplitude=self.swing_amplitude,
            swing_speed_scale=self.swing_speed_scale,
            com_shift_y=self.com_shift_y,
            pelvis_yaw_offset=self.pelvis_yaw_offset,
            foot_yaw_offset=self.foot_yaw_offset,
            foot_pitch_offset=self.foot_pitch_offset,
            loft_synergy=self.loft_synergy,
            policy_type="parameter",
        )

    @property
    def candidate_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def parameter_overrides(self) -> dict[str, Any]:
        return {
            "stance_offset_x": self.stance_offset_x,
            "stance_offset_y": self.stance_offset_y,
            "kick_foot": self.kick_foot,
            "swing_amplitude": self.swing_amplitude,
            "swing_speed_scale": self.swing_speed_scale,
            "com_shift_y": self.com_shift_y,
            "pelvis_yaw_offset": self.pelvis_yaw_offset,
            "foot_yaw_offset": self.foot_yaw_offset,
            "foot_pitch_offset": self.foot_pitch_offset,
            "loft_synergy": self.loft_synergy,
        }


def _roll_pitch_deg(quaternion_wxyz: NDArray[np.float64]) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return math.degrees(roll), math.degrees(math.asin(sin_pitch))


def _direction_error_deg(velocity_xy: NDArray[np.float64], target_deg: float) -> float:
    speed = float(np.linalg.norm(velocity_xy))
    if speed <= 1.0e-9:
        return 180.0
    actual = math.degrees(math.atan2(float(velocity_xy[1]), float(velocity_xy[0])))
    return abs((actual - target_deg + 180.0) % 360.0 - 180.0)


def _first_stable_action_latency(
    *,
    time: NDArray[np.float64],
    velocity: NDArray[np.float64],
    first_contact_index: int,
    target_direction_deg: float,
    gate: FirstTouchGateConfig,
) -> float:
    passing: list[bool] = []
    for index in range(first_contact_index, time.size):
        latency = float(time[index] - time[first_contact_index])
        if latency > gate.maximum_next_action_latency_sec:
            break
        speed = float(np.linalg.norm(velocity[index, :2]))
        passing.append(
            gate.minimum_outgoing_speed_mps <= speed <= gate.maximum_outgoing_speed_mps
            and _direction_error_deg(velocity[index, :2], target_direction_deg)
            <= gate.maximum_direction_error_deg
        )
        if len(passing) >= 3 and all(passing[-3:]):
            return max(0.0, latency)
    return gate.maximum_next_action_latency_sec + 0.01


def measure_first_touch_trajectory(
    *,
    trace: dict[str, NDArray[Any]],
    result: G1SharedWorldResult,
    scenario: FirstTouchPhysicsScenario,
    candidate: FirstTouchCandidate,
    qualification: G1AssetQualification,
    gate: FirstTouchGateConfig | None = None,
) -> tuple[FirstTouchMeasurement, FirstTouchEvaluation]:
    """Convert physics arrays into the task-level causal measurement."""

    active_gate = gate or FirstTouchGateConfig()
    required = {
        "time",
        "ball_pose",
        "ball_velocity",
        "ball_contact_role",
        "shooter_ball_contact_foot",
        "shooter_pelvis_pose",
        "shooter_torso_quaternion",
    }
    if required - trace.keys():
        raise ValueError("First Touch trajectory is missing required physics arrays")
    time = np.asarray(trace["time"], dtype=np.float64)
    ball_pose = np.asarray(trace["ball_pose"], dtype=np.float64)
    ball_velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
    contact_role = np.asarray(trace["ball_contact_role"], dtype=np.int64)
    contact_foot = np.asarray(trace["shooter_ball_contact_foot"], dtype=np.int64)
    pelvis = np.asarray(trace["shooter_pelvis_pose"], dtype=np.float64)
    torso = np.asarray(trace["shooter_torso_quaternion"], dtype=np.float64)
    length = time.size
    if (
        length < 2
        or ball_pose.shape != (length, 7)
        or ball_velocity.shape != (length, 6)
        or contact_role.shape != (length,)
        or contact_foot.shape != (length,)
        or pelvis.shape != (length, 7)
        or torso.shape != (length, 4)
        or not all(
            np.all(np.isfinite(value)) for value in (time, ball_pose, ball_velocity, pelvis, torso)
        )
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("First Touch trajectory arrays are invalid")

    shooter_contact = np.flatnonzero(contact_role == 2)
    first_shooter_contact = int(shooter_contact[0]) if shooter_contact.size else -1
    contact_detected = bool(shooter_contact.size and int(contact_foot[first_shooter_contact]) != 0)
    first_contact = first_shooter_contact if contact_detected else -1
    precontact = max(0, first_contact - 1) if contact_detected else 0
    incoming_speed = float(np.linalg.norm(ball_velocity[precontact, :2]))
    if not contact_detected:
        incoming_speed = float(np.linalg.norm(ball_velocity[0, :2]))

    if contact_detected:
        horizon_time = float(time[first_contact] + scenario.measurement_horizon_sec)
        horizon = min(length - 1, int(np.searchsorted(time, horizon_time, side="left")))
        outgoing_velocity = ball_velocity[horizon, :2]
        outgoing_speed = float(np.linalg.norm(outgoing_velocity))
        angle_rad = math.radians(scenario.target_direction_deg)
        target_unit = np.asarray((math.cos(angle_rad), math.sin(angle_rad)), dtype=np.float64)
        desired_position = (
            ball_pose[first_contact, :2]
            + target_unit * scenario.target_outgoing_speed_mps * scenario.measurement_horizon_sec
        )
        target_error = float(np.linalg.norm(ball_pose[horizon, :2] - desired_position))
        direction_error = _direction_error_deg(outgoing_velocity, scenario.target_direction_deg)
        latency = _first_stable_action_latency(
            time=time,
            velocity=ball_velocity,
            first_contact_index=first_contact,
            target_direction_deg=scenario.target_direction_deg,
            gate=active_gate,
        )
        selected_foot = "left" if int(contact_foot[first_contact]) < 0 else "right"
    else:
        outgoing_speed = 0.0
        target_error = max(active_gate.maximum_target_error_m + 0.01, 1.0)
        direction_error = 180.0
        latency = active_gate.maximum_next_action_latency_sec + 0.01
        selected_foot = candidate.kick_foot

    dt = np.diff(time)
    root_velocity = np.diff(pelvis[:, :3], axis=0) / dt[:, None]
    maximum_root_speed = float(np.linalg.norm(root_velocity, axis=1).max())
    tilts = [_roll_pitch_deg(row) for row in torso]
    maximum_torso_tilt = max(max(abs(roll), abs(pitch)) for roll, pitch in tilts)
    snapshot_hash = hash_json(
        {
            "scenario_hash": scenario.scenario_hash,
            "candidate_hash": candidate.candidate_hash,
            "body_hash": qualification.body_hash,
            "kick_prior_hash": qualification.kick_prior_hash,
        }
    )
    measurement = FirstTouchMeasurement(
        sample_id=f"touch.{scenario.scenario_id}.{candidate.candidate_id}",
        actor_id="soccer.g1.first_touch_receiver",
        source_snapshot_hash=snapshot_hash,
        body_hash=qualification.body_hash,
        scenario_hash=scenario.scenario_hash,
        incoming_speed_mps=incoming_speed,
        outgoing_speed_mps=outgoing_speed,
        target_error_m=target_error,
        direction_error_deg=direction_error,
        next_action_latency_sec=latency,
        minimum_pelvis_height_m=float(pelvis[:, 2].min()),
        maximum_torso_tilt_deg=maximum_torso_tilt,
        maximum_root_speed_mps=maximum_root_speed,
        contact_detected=contact_detected,
        selected_foot=selected_foot,
        required_foot=candidate.kick_foot,
    )
    if (
        not result.finite_state
        or result.joint_limit_violation
        or result.torque_limit_violation
        or result.shooter_post_kick_fall
    ):
        measurement = FirstTouchMeasurement(
            **{
                **asdict(measurement),
                "minimum_pelvis_height_m": min(
                    measurement.minimum_pelvis_height_m,
                    active_gate.minimum_pelvis_height_m - 0.01,
                ),
            }
        )
    return measurement, evaluate_first_touch(measurement, active_gate)


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_first_touch_physics_case(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    scenario: FirstTouchPhysicsScenario,
    candidate: FirstTouchCandidate,
    gate: FirstTouchGateConfig | None = None,
) -> dict[str, Any]:
    """Execute one G1 moving-ball case and persist content-bound evidence."""

    destination = output_dir.expanduser().resolve()
    source = source_checkout.expanduser().resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("First Touch evidence output must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    goal = G1TrainingGoalSpec(
        plane_x_m=12.0,
        target_y_m=0.8,
        target_z_m=1.0,
    )
    result, trace = simulate_shared_world(
        qualification.asset_root,
        shooter_start_sec=candidate.receiver_start_delay_sec,
        shooter_target=(goal.plane_x_m, goal.target_y_m, goal.target_z_m),
        shooter_policy_target=scenario.frozen_policy_target_m,
        shooter_parameter_overrides=candidate.parameter_overrides(),
        ball_launcher_position_m=scenario.launcher_position_m,
        ball_launcher_velocity_mps=scenario.launcher_velocity_mps,
        launcher_receiver_enabled=True,
        receiver_phase_sync_enabled=True,
        shooter_joint_guard_enabled=True,
        shooter_precontact_joint_guard_enabled=True,
        shooter_joint_guard_config=G1JointGuardConfig(
            margin_rad=0.10,
            prediction_horizon_sec=0.20,
            boundary_kp=200.0,
            boundary_kd=20.0,
        ),
        shooter_post_policy_frame=260,
        shooter_post_policy_blend_frames=0,
        shooter_post_policy_recovery_enabled=True,
        passer_origin=scenario.frozen_partner_origin_m,
        goal_spec=goal,
        simulation_duration_sec=scenario.simulation_duration_sec,
    )
    measurement, evaluation = measure_first_touch_trajectory(
        trace=trace,
        result=result,
        scenario=scenario,
        candidate=candidate,
        qualification=qualification,
        gate=gate,
    )
    destination.mkdir(parents=True)
    trajectory_path = destination / "trajectory.npz"
    temporary_trajectory = destination / "trajectory.npz.tmp"
    with temporary_trajectory.open("wb") as stream:
        np.savez_compressed(stream, **trace)  # type: ignore[arg-type]
    os.replace(temporary_trajectory, trajectory_path)
    active_gate = gate or FirstTouchGateConfig()
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.first_touch_physics_evidence.v1",
        "status": "PASS_FIRST_TOUCH" if evaluation.passed else "FAILURE_ATTRIBUTED",
        "scenario": asdict(scenario),
        "scenario_hash": scenario.scenario_hash,
        "candidate": asdict(candidate),
        "candidate_hash": candidate.candidate_hash,
        "gate": asdict(active_gate),
        "measurement": asdict(measurement),
        "measurement_hash": measurement.measurement_hash,
        "evaluation": evaluation.to_dict(),
        "evaluation_hash": evaluation.evaluation_hash,
        "physics": {
            "authority": "CPU_MUJOCO",
            "strict_replay": True,
            "pixels_used_for_scoring": False,
            "physics_steps": result.physics_steps,
            "trajectory_digest": trajectory_digest(trace),
            "trajectory_artifact": trajectory_path.name,
            "trajectory_artifact_hash": hash_bytes(trajectory_path.read_bytes()),
            "finite_state": result.finite_state,
            "joint_limit_violation": result.joint_limit_violation,
            "torque_limit_violation": result.torque_limit_violation,
            "post_touch_fall": result.shooter_post_kick_fall,
        },
        "provenance": {
            "source_commit": _git_head(source),
            "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
            "body_hash": qualification.body_hash,
            "kick_prior_hash": qualification.kick_prior_hash,
            "motion_hash": qualification.motion_hash,
            "backend_commit": qualification.backend_commit,
        },
        "evidence_ceiling": {
            "activation_ceiling": "SIM_ONLY",
            "g1_policy_executed": True,
            "single_case_only": True,
            "promotion_eligible": False,
            "statement": (
                "One CPU MuJoCo acquisition case cannot promote a First Touch policy; "
                "balanced acquisition and retention suites are still required."
            ),
        },
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "first-touch-report.json", report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--scenario-id", default="s118a.baseline")
    parser.add_argument("--candidate-id", default="frozen-prior")
    parser.add_argument("--incoming-speed", type=float, default=0.9)
    parser.add_argument("--minimum-measured-incoming-speed", type=float, default=0.5)
    parser.add_argument("--incoming-lateral", type=float, default=0.0)
    parser.add_argument("--target-direction", type=float, default=0.0)
    parser.add_argument("--target-speed", type=float, default=1.2)
    parser.add_argument("--kick-foot", choices=("left", "right"), default="right")
    parser.add_argument("--receiver-start-delay", type=float, default=0.0)
    parser.add_argument("--stance-offset-x", type=float, default=0.0)
    parser.add_argument("--stance-offset-y", type=float, default=-0.06)
    parser.add_argument("--swing-amplitude", type=float, default=1.0)
    parser.add_argument("--swing-speed-scale", type=float, default=0.90)
    parser.add_argument("--com-shift-y", type=float, default=-0.065)
    parser.add_argument("--pelvis-yaw-offset", type=float, default=0.175)
    parser.add_argument("--foot-yaw-offset", type=float, default=0.03025)
    parser.add_argument("--foot-pitch-offset", type=float, default=0.0)
    parser.add_argument("--loft-synergy", type=float, default=0.0)
    args = parser.parse_args()
    report = run_first_touch_physics_case(
        asset_root=args.asset_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        scenario=FirstTouchPhysicsScenario(
            scenario_id=args.scenario_id,
            incoming_speed_mps=args.incoming_speed,
            incoming_lateral_m=args.incoming_lateral,
            target_direction_deg=args.target_direction,
            target_outgoing_speed_mps=args.target_speed,
        ),
        candidate=FirstTouchCandidate(
            candidate_id=args.candidate_id,
            kick_foot=args.kick_foot,
            receiver_start_delay_sec=args.receiver_start_delay,
            stance_offset_x=args.stance_offset_x,
            stance_offset_y=args.stance_offset_y,
            swing_amplitude=args.swing_amplitude,
            swing_speed_scale=args.swing_speed_scale,
            com_shift_y=args.com_shift_y,
            pelvis_yaw_offset=args.pelvis_yaw_offset,
            foot_yaw_offset=args.foot_yaw_offset,
            foot_pitch_offset=args.foot_pitch_offset,
            loft_synergy=args.loft_synergy,
        ),
        gate=FirstTouchGateConfig(
            minimum_incoming_speed_mps=args.minimum_measured_incoming_speed,
        ),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "FirstTouchCandidate",
    "FirstTouchPhysicsScenario",
    "measure_first_touch_trajectory",
    "run_first_touch_physics_case",
]
