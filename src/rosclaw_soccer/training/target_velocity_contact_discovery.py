"""Teacher data collection for target-conditioned G1 contact control."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.shoot.loft_teacher import G1LoftTeacherConfig
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _chain_quality,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)


@dataclass(frozen=True)
class TargetVelocityContactProbe:
    context: CausalTransitionContext
    target_foot_velocity_xyz_mps: tuple[float, float, float]
    maximum_arrival_advance_frames: int
    stance_offset_x_m: float
    stance_offset_y_m: float
    contact_policy_frame: int = 248
    foot_yaw_offset_rad: float = 0.04
    foot_pitch_offset_rad: float = 0.01
    maximum_teacher_force_xyz_n: tuple[float, float, float] = (80.0, 80.0, 60.0)

    def __post_init__(self) -> None:
        target = np.asarray(self.target_foot_velocity_xyz_mps, dtype=np.float64)
        force = np.asarray(self.maximum_teacher_force_xyz_n, dtype=np.float64)
        values = (
            self.stance_offset_x_m,
            self.stance_offset_y_m,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
        )
        if (
            target.shape != (3,)
            or force.shape != (3,)
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(force))
            or not 5.0 <= target[0] <= 12.0
            or (target[1] != 0.0 and not 1.0 <= abs(target[1]) <= 6.0)
            or (target[2] != 0.0 and not (-3.0 <= target[2] <= -0.5 or 3.0 <= target[2] <= 6.0))
            or any(not math.isfinite(value) for value in values)
            or self.maximum_arrival_advance_frames not in {0, 6, 12, 18}
            or not -0.12 <= self.stance_offset_x_m <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not 238 <= self.contact_policy_frame <= 258
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.08 <= self.foot_pitch_offset_rad <= 0.08
            or np.any(force < 10.0)
            or np.any(force > 200.0)
        ):
            raise ValueError("target-velocity probe exceeds its SIM-only envelope")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def teacher_config(self) -> G1LoftTeacherConfig:
        target = self.target_foot_velocity_xyz_mps
        return G1LoftTeacherConfig(
            target_forward_speed_mps=target[0],
            target_lateral_speed_mps=target[1],
            target_vertical_speed_mps=target[2],
            maximum_forward_force_n=self.maximum_teacher_force_xyz_n[0],
            maximum_lateral_force_n=self.maximum_teacher_force_xyz_n[1],
            maximum_vertical_force_n=self.maximum_teacher_force_xyz_n[2],
            maximum_foot_ball_distance_m=0.50,
        )


def run_target_velocity_contact_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_path: Path,
    probes: tuple[TargetVelocityContactProbe, ...],
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    if len(probes) < 8 or len({probe.probe_hash for probe in probes}) != len(probes):
        raise ValueError("target-velocity discovery needs eight unique probes")
    if not 1 <= workers <= 8:
        raise ValueError("target-velocity discovery workers must be in [1, 8]")
    targets = np.asarray([probe.target_foot_velocity_xyz_mps for probe in probes], dtype=np.float64)
    if any(np.ptp(targets[:, axis]) <= 1.0e-9 for axis in range(3)):
        raise ValueError("target-velocity discovery must vary all three target axes")
    rejected_path = rejected_exam_path.expanduser().resolve()
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    claimed = rejected.pop("report_hash", None)
    if (
        claimed != hash_json(rejected)
        or rejected.get("status") != "REJECTED_DUAL_CLOCK_CONTACT_RETENTION"
        or rejected.get("promotion_eligible") is not False
        or rejected.get("sealed") is not True
    ):
        raise ValueError("target-velocity discovery requires intact failure memory")
    rejected_contexts = {str(row["context_hash"]) for row in rejected["rows"]}
    if not {probe.context.context_hash for probe in probes} <= rejected_contexts:
        raise ValueError("target-velocity probes must replay the consumed failure partition")

    quality = quality_config or CausalTransitionGrowthConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    implementation_hash = _implementation_hash()
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_velocity_contact_discovery_request.v1",
        "partition": "CONSUMED_REJECTED_HOLDOUT_TEACHER_DEVELOPMENT",
        "probes": [asdict(probe) for probe in probes],
        "probe_hashes": [probe.probe_hash for probe in probes],
        "rejected_exam_report_hash": claimed,
        "rejected_exam_file_hash": hash_bytes(rejected_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "teacher_role": "SIM_ONLY_DATA_GENERATOR",
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            output,
            index,
            probe,
            quality,
        )
        for index, probe in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    success_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    active_count = sum(bool(row["teacher_active"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_velocity_contact_discovery.v1",
        "status": (
            "PASS_TARGET_VELOCITY_CONTACT_DISCOVERY"
            if success_count >= 1 and safe_count >= len(rows) - 2 and active_count >= 8
            else "REJECTED_TARGET_VELOCITY_CONTACT_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_HOLDOUT_TEACHER_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_report_hash": claimed,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "metrics": {
            "probe_count": len(rows),
            "chain_success_count": success_count,
            "safe_count": safe_count,
            "teacher_active_count": active_count,
        },
        "rows": rows,
        "implementation_hash": implementation_hash,
        "teacher_role": "SIM_ONLY_DATA_GENERATOR",
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "discovery-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        int,
        TargetVelocityContactProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, output, index, probe, quality = job
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=probe.context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    kwargs.update(
        shooter_causal_strike_option_config=replace(
            G1CausalStrikeOptionConfig(),
            maximum_arrival_advance_frames=probe.maximum_arrival_advance_frames,
        ),
        shooter_ballistic_contact_torque_config=replace(
            UpperCornerStrikePolicy().torque_config(),
            contact_policy_frame=probe.contact_policy_frame,
        ),
        shooter_loft_teacher_config=probe.teacher_config(),
        shooter_precontact_joint_guard_enabled=True,
        shooter_parameter_overrides={
            "stance_offset_x": probe.stance_offset_x_m,
            "stance_offset_y": probe.stance_offset_y_m,
            "foot_yaw_offset": probe.foot_yaw_offset_rad,
            "foot_pitch_offset": probe.foot_pitch_offset_rad,
        },
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe": asdict(probe),
        "probe_hash": probe.probe_hash,
        "teacher_config_hash": probe.teacher_config().config_hash,
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "teacher_active": bool(np.any(active)),
        "teacher_active_frame_count": int(np.count_nonzero(active)),
        "trajectory": artifact,
    }


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "shoot" / "loft_teacher.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    )
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("target-velocity discovery output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["TargetVelocityContactProbe", "run_target_velocity_contact_discovery"]
