"""CPU-MuJoCo discovery for a teacher-distilled three-axis strike actor."""

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
class ThreeAxisContactProbe:
    """One bounded contact mode evaluated against a complete incoming pass."""

    context: CausalTransitionContext
    maximum_arrival_advance_frames: int
    stance_offset_x_m: float
    stance_offset_y_m: float = -0.06
    contact_policy_frame: int = 248

    def __post_init__(self) -> None:
        values = (self.stance_offset_x_m, self.stance_offset_y_m)
        if (
            any(not math.isfinite(value) for value in values)
            or self.maximum_arrival_advance_frames not in {0, 12}
            or not -0.12 <= self.stance_offset_x_m <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not 238 <= self.contact_policy_frame <= 258
        ):
            raise ValueError("three-axis contact probe exceeds its SIM-only envelope")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_three_axis_contact_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    probes: tuple[ThreeAxisContactProbe, ...],
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    teacher_config: G1LoftTeacherConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Collect immutable success/failure trajectories with an isolated teacher."""

    if len(probes) < 4 or len({probe.probe_hash for probe in probes}) != len(probes):
        raise ValueError("three-axis discovery requires four unique probes")
    if not 1 <= workers <= 6:
        raise ValueError("three-axis discovery workers must be in [1, 6]")
    quality = quality_config or CausalTransitionGrowthConfig()
    teacher = teacher_config or G1LoftTeacherConfig(
        target_forward_speed_mps=5.0,
        maximum_foot_ball_distance_m=0.50,
    )
    if not teacher.enabled:
        raise ValueError("three-axis discovery requires an enabled SIM teacher")
    output = _new_external_output(output_dir)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    _, source = _load_lead_policy(source_s95_dir)
    implementation_hash = _implementation_hash()
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.three_axis_contact_discovery_request.v1",
        "probes": [asdict(probe) for probe in probes],
        "probe_hashes": [probe.probe_hash for probe in probes],
        "teacher_config": asdict(teacher),
        "teacher_config_hash": teacher.config_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "source_s95_evidence_hash": source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
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
            teacher,
        )
        for index, probe in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    chain_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    active_count = sum(bool(row["teacher_active"]) for row in rows)
    discovery_passed = chain_count >= 1 and safe_count >= len(rows) - 1 and active_count >= 4
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.three_axis_contact_discovery.v1",
        "status": (
            "PASS_THREE_AXIS_CONTACT_DISCOVERY"
            if discovery_passed
            else "REJECTED_THREE_AXIS_CONTACT_DISCOVERY"
        ),
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "teacher_config": asdict(teacher),
        "metrics": {
            "probe_count": len(rows),
            "chain_success_count": chain_count,
            "safe_count": safe_count,
            "teacher_active_count": active_count,
        },
        "rows": rows,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
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
        ThreeAxisContactProbe,
        CausalTransitionGrowthConfig,
        G1LoftTeacherConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, output, index, probe, quality, teacher = job
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
        shooter_loft_teacher_config=teacher,
        shooter_precontact_joint_guard_enabled=True,
        shooter_parameter_overrides={
            "stance_offset_x": probe.stance_offset_x_m,
            "stance_offset_y": probe.stance_offset_y_m,
            "foot_yaw_offset": 0.04,
            "foot_pitch_offset": 0.01,
        },
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe": asdict(probe),
        "probe_hash": probe.probe_hash,
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "teacher_active": bool(np.any(active)),
        "teacher_active_frame_count": int(np.count_nonzero(active)),
        "trajectory": artifact,
    }


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "three_axis_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "shoot" / "loft_teacher.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    )
    return str(hash_json({path.name: hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("three-axis discovery output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["ThreeAxisContactProbe", "run_three_axis_contact_discovery"]
