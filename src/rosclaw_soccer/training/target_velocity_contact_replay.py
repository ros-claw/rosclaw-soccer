"""Teacher-free physical replay of a target-conditioned contact actor."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.target_velocity_contact_actor import (
    load_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _chain_quality,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.target_velocity_contact_discovery import (
    TargetVelocityContactProbe,
)


def run_target_velocity_contact_replay(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    discovery_report_path: Path,
    actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("target-velocity replay workers must be in [1, 8]")
    source_path = discovery_report_path.expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    claimed = source.pop("report_hash", None)
    if (
        claimed != hash_json(source)
        or source.get("schema_version") != "rosclaw.growth.target_velocity_contact_discovery.v1"
        or source.get("status") != "PASS_TARGET_VELOCITY_CONTACT_DISCOVERY"
        or source.get("promotion_eligible") is not False
    ):
        raise ValueError("target-velocity replay requires passing discovery data")
    source["report_hash"] = claimed
    actor = load_g1_target_velocity_contact_actor(actor_path)
    if claimed not in actor.source_evidence_hashes:
        raise ValueError("target-velocity actor is not bound to its replay source")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if actor.body_hash != qualification.body_hash:
        raise ValueError("target-velocity replay Body identity changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_velocity_contact_replay_request.v1",
        "source_discovery_report_hash": claimed,
        "source_discovery_file_hash": hash_bytes(source_path.read_bytes()),
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
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
            actor_path.expanduser().resolve(),
            output,
            index,
            _probe_from_dict(row["probe"]),
            bool(row["quality"]["chain_passed"]),
            quality,
        )
        for index, row in enumerate(source["rows"])
    )
    if workers == 1:
        rows = [_run_replay(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_replay, jobs))
    success_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    teacher_success_count = sum(bool(row["teacher_chain_passed"]) for row in rows)
    retained_teacher_success_count = sum(
        bool(row["teacher_chain_passed"] and row["quality"]["chain_passed"]) for row in rows
    )
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    goal_count = sum(bool(row["result"]["goal_crossed"]) for row in rows)
    save_count = sum(bool(row["result"]["goalkeeper_save_observed"]) for row in rows)
    gates = {
        "all_safe": safe_count == len(rows),
        "teacher_absent": all(not bool(row["teacher_active"]) for row in rows),
        "actor_executed": all(bool(row["actor_active"]) for row in rows),
        "retains_teacher_successes": retained_teacher_success_count
        >= max(1, teacher_success_count - 1),
        "minimum_chain_success": success_count >= 4,
        "both_goal_and_save": goal_count >= 1 and save_count >= 1,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_velocity_contact_replay.v1",
        "status": (
            "PASS_TARGET_VELOCITY_CONTACT_REPLAY"
            if all(gates.values())
            else "REJECTED_TARGET_VELOCITY_CONTACT_REPLAY"
        ),
        "promotion_eligible": False,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_discovery_report_hash": claimed,
        "actor_hash": actor.actor_hash,
        "metrics": {
            "case_count": len(rows),
            "chain_success_count": success_count,
            "teacher_chain_success_count": teacher_success_count,
            "retained_teacher_success_count": retained_teacher_success_count,
            "safe_count": safe_count,
            "goal_count": goal_count,
            "save_count": save_count,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "replay-report.json", report)
    return report


def _run_replay(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        TargetVelocityContactProbe,
        bool,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, actor_path, output, index, probe, teacher_passed, quality = job
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
        shooter_target_velocity_contact_actor_path=actor_path,
        shooter_target_foot_velocity_xyz_mps=probe.target_foot_velocity_xyz_mps,
        shooter_precontact_joint_guard_enabled=True,
        shooter_parameter_overrides={
            "stance_offset_x": probe.stance_offset_x_m,
            "stance_offset_y": probe.stance_offset_y_m,
            "foot_yaw_offset": probe.foot_yaw_offset_rad,
            "foot_pitch_offset": probe.foot_pitch_offset_rad,
        },
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"case-{index:03d}.npz", trajectory)
    actor_active = np.asarray(
        trajectory["shooter_target_velocity_contact_actor_active"], dtype=np.bool_
    )
    teacher_active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "case_index": index,
        "probe": asdict(probe),
        "probe_hash": probe.probe_hash,
        "teacher_chain_passed": teacher_passed,
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "actor_active": bool(np.any(actor_active)),
        "actor_active_frame_count": int(np.count_nonzero(actor_active)),
        "teacher_active": bool(np.any(teacher_active)),
        "trajectory": artifact,
    }


def _probe_from_dict(payload: dict[str, Any]) -> TargetVelocityContactProbe:
    context = dict(payload["context"])
    context["passer_origin_m"] = tuple(context["passer_origin_m"])
    context["passer_ball_local_xy_m"] = tuple(context["passer_ball_local_xy_m"])
    values = dict(payload)
    values["context"] = CausalTransitionContext(**context)
    values["target_foot_velocity_xyz_mps"] = tuple(values["target_foot_velocity_xyz_mps"])
    values["maximum_teacher_force_xyz_n"] = tuple(values["maximum_teacher_force_xyz_n"])
    return TargetVelocityContactProbe(**values)


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    )
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("target-velocity replay output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["run_target_velocity_contact_replay"]
