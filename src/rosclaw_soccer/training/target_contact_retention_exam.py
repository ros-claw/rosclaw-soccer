"""Fresh sealed retention exam for coherent target-conditioned contact."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.planned_contact_mode_actor import (
    load_planned_contact_mode_actor,
    planned_contact_mode_features,
)
from rosclaw_soccer.growth.runtime_contact_mode_actor import load_runtime_contact_mode_actor
from rosclaw_soccer.growth.target_contact_plan_actor import load_target_contact_plan_actor
from rosclaw_soccer.growth.target_velocity_contact_actor import (
    load_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.growth.three_axis_contact_actor import load_g1_three_axis_contact_actor
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


def default_target_contact_holdouts() -> tuple[CausalTransitionContext, ...]:
    """S129 v5 contexts registered before their first physics rollout."""

    origin = (5.10, -0.16406006503921598, 0.0)
    return (
        CausalTransitionContext(
            "s129.holdout.v5.00", origin, -0.1145, 1.3435, (1.1885, -0.1485), 0.80, 0.0887
        ),
        CausalTransitionContext(
            "s129.holdout.v5.01", origin, -0.1155, 1.3445, (1.1875, -0.1475), 0.80, 0.0883
        ),
        CausalTransitionContext(
            "s129.holdout.v5.02", origin, -0.1140, 1.3450, (1.1890, -0.1490), 0.80, 0.0890
        ),
        CausalTransitionContext(
            "s129.holdout.v5.03", origin, -0.0865, 1.2995, (1.2125, -0.1685), 0.80, 0.1018
        ),
        CausalTransitionContext(
            "s129.holdout.v5.04", origin, -0.0875, 1.3005, (1.2115, -0.1675), 0.80, 0.1022
        ),
        CausalTransitionContext(
            "s129.holdout.v5.05", origin, -0.0860, 1.3010, (1.2130, -0.1690), 0.80, 0.1015
        ),
    )


def run_target_contact_retention_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    target_plan_actor_path: Path,
    target_contact_actor_path: Path,
    frozen_parent_plan_actor_path: Path,
    frozen_parent_runtime_actor_path: Path,
    frozen_parent_contact_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    holdouts: tuple[CausalTransitionContext, ...] | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    contexts = holdouts or default_target_contact_holdouts()
    if len(contexts) != 6 or len({context.context_hash for context in contexts}) != 6:
        raise ValueError("target contact retention needs six unique holdouts")
    if not 1 <= workers <= 6:
        raise ValueError("target contact retention workers must be in [1, 6]")
    quality = quality_config or CausalTransitionGrowthConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    plan = load_target_contact_plan_actor(target_plan_actor_path)
    contact = load_g1_target_velocity_contact_actor(target_contact_actor_path)
    parent_plan = load_planned_contact_mode_actor(frozen_parent_plan_actor_path)
    parent_runtime = load_runtime_contact_mode_actor(frozen_parent_runtime_actor_path)
    parent_contact = load_g1_three_axis_contact_actor(frozen_parent_contact_actor_path)
    if (
        {
            plan.body_hash,
            contact.body_hash,
            parent_plan.body_hash,
            parent_runtime.body_hash,
            parent_contact.body_hash,
        }
        != {qualification.body_hash}
        or plan.target_contact_actor_hash != contact.actor_hash
        or plan.kick_prior_hash != qualification.kick_prior_hash
        or parent_plan.kick_prior_hash != qualification.kick_prior_hash
        or parent_runtime.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("target contact retention asset lineage changed")
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    implementation_hash = _implementation_hash()
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_contact_retention_request.v1",
        "partition": "SEALED_HOLDOUT",
        "sealed": True,
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "target_plan_actor_hash": plan.actor_hash,
        "target_plan_actor_file_hash": hash_bytes(target_plan_actor_path.read_bytes()),
        "target_contact_actor_hash": contact.actor_hash,
        "target_contact_actor_file_hash": hash_bytes(target_contact_actor_path.read_bytes()),
        "frozen_parent_plan_actor_hash": parent_plan.actor_hash,
        "frozen_parent_runtime_actor_hash": parent_runtime.actor_hash,
        "frozen_parent_contact_actor_hash": parent_contact.actor_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "source_s95_evidence_hash": source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
        "late_stance_rewrite_allowed": False,
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            target_plan_actor_path.expanduser().resolve(),
            target_contact_actor_path.expanduser().resolve(),
            frozen_parent_plan_actor_path.expanduser().resolve(),
            frozen_parent_runtime_actor_path.expanduser().resolve(),
            frozen_parent_contact_actor_path.expanduser().resolve(),
            output,
            index,
            context,
            quality,
        )
        for index, context in enumerate(contexts)
    )
    if workers == 1:
        rows = [_run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, jobs))
    candidate_success = sum(bool(row["candidate_quality"]["chain_passed"]) for row in rows)
    parent_success = sum(bool(row["parent_quality"]["chain_passed"]) for row in rows)
    goal_count = sum(bool(row["candidate_result"]["goal_crossed"]) for row in rows)
    save_count = sum(bool(row["candidate_result"]["goalkeeper_save_observed"]) for row in rows)
    metrics = {
        "case_count": len(rows),
        "candidate_chain_success_count": candidate_success,
        "parent_chain_success_count": parent_success,
        "chain_success_gain": candidate_success - parent_success,
        "candidate_safe_rate": sum(bool(row["candidate_quality"]["safe"]) for row in rows)
        / len(rows),
        "parent_safe_rate": sum(bool(row["parent_quality"]["safe"]) for row in rows) / len(rows),
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in rows) / len(rows),
        "plan_accept_count": sum(bool(row["plan_decision"]["accepted"]) for row in rows),
        "goal_count": goal_count,
        "save_count": save_count,
        "mean_candidate_shot_speed_mps": float(
            np.mean([row["candidate_result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
    }
    gates = {
        "plan_accepts_all": metrics["plan_accept_count"] == len(rows),
        "candidate_safe_rate": metrics["candidate_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "minimum_candidate_success": candidate_success >= 4,
        "measured_gain_over_parent": candidate_success >= parent_success + 1,
        "both_support_clusters_succeed": any(
            row["candidate_quality"]["chain_passed"] for row in rows[:3]
        )
        and any(row["candidate_quality"]["chain_passed"] for row in rows[3:]),
        "both_goal_and_save": goal_count >= 1 and save_count >= 1,
        "measured_arrival_confirmed": all(row["measured_arrival_confirmed"] for row in rows),
        "teacher_absent": all(row["teacher_active_fraction"] == 0.0 for row in rows),
        "target_contact_actor_executed": all(
            row["target_actor_active_fraction"] > 0.0 for row in rows
        ),
        "late_stance_rewrite_absent": True,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.target_contact_retention.v1",
        "status": (
            "PASS_TARGET_CONTACT_RETENTION" if passed else "REJECTED_TARGET_CONTACT_RETENTION"
        ),
        "sealed": True,
        "partition": "SEALED_HOLDOUT",
        "promotion_eligible": passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "target_plan_actor_hash": plan.actor_hash,
        "target_contact_actor_hash": contact.actor_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": implementation_hash,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "world_reset_after_pass_or_shot": False,
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "precontact_plan_uses_only_registered_task_context": True,
            "arrival_confirmation_uses_measured_ball_state": True,
            "late_stance_rewrite_allowed": False,
            "teacher_enabled": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "retention-report.json", report)
    return report


def _run_case(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        plan_path,
        contact_path,
        parent_plan_path,
        parent_runtime_path,
        parent_contact_path,
        output,
        index,
        context,
        quality,
    ) = job
    features = planned_contact_mode_features(
        receiver_lane_m=context.receiver_lane_m,
        reception_target_x_m=context.reception_target_x_m,
        passer_ball_local_xy_m=context.passer_ball_local_xy_m,
        ball_ground_friction=context.ball_ground_friction,
    )
    plan = load_target_contact_plan_actor(plan_path).decide(features)
    parent_plan = load_planned_contact_mode_actor(parent_plan_path).decide(features)
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    candidate_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    if plan.action is not None:
        action = plan.action
        candidate_kwargs.update(
            shooter_parameter_overrides={
                "stance_offset_x": action.stance_offset_x_m,
                "stance_offset_y": action.stance_offset_y_m,
                "foot_yaw_offset": action.foot_yaw_offset_rad,
                "foot_pitch_offset": action.foot_pitch_offset_rad,
            },
            shooter_causal_strike_option_config=replace(
                G1CausalStrikeOptionConfig(),
                maximum_arrival_advance_frames=action.maximum_arrival_advance_frames,
            ),
            shooter_ballistic_contact_torque_config=replace(
                UpperCornerStrikePolicy().torque_config(),
                contact_policy_frame=action.contact_policy_frame,
            ),
            shooter_target_velocity_contact_actor_path=contact_path,
            shooter_target_foot_velocity_xyz_mps=action.target_foot_velocity_xyz_mps,
            shooter_precontact_joint_guard_enabled=True,
        )
    parent_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    if parent_plan.action is not None:
        parent_action = parent_plan.action
        parent_kwargs.update(
            shooter_parameter_overrides={
                "stance_offset_x": parent_action.stance_offset_x_m,
                "stance_offset_y": parent_action.stance_offset_y_m,
                "foot_yaw_offset": parent_action.foot_yaw_offset_rad,
                "foot_pitch_offset": parent_action.foot_pitch_offset_rad,
            },
            shooter_causal_strike_option_config=G1CausalStrikeOptionConfig(),
            shooter_runtime_contact_mode_actor_path=parent_runtime_path,
            shooter_ballistic_contact_torque_config=replace(
                UpperCornerStrikePolicy().torque_config(),
                contact_policy_frame=action.contact_policy_frame,
            ),
            shooter_three_axis_contact_actor_path=parent_contact_path,
            shooter_precontact_joint_guard_enabled=True,
        )
    candidate_result, candidate_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    candidate_artifact = _save_trajectory(case_dir / "candidate-primary.npz", candidate_trajectory)
    replay_artifact = _save_trajectory(case_dir / "candidate-replay.npz", replay_trajectory)
    parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trajectory)
    incoming = np.asarray(
        candidate_trajectory["shooter_causal_strike_option_incoming_observation_count"],
        dtype=np.int64,
    )
    bridge = np.asarray(
        candidate_trajectory["shooter_causal_strike_option_begin_bridge"], dtype=np.bool_
    )
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "plan_decision": asdict(plan),
        "parent_plan_decision": asdict(parent_plan),
        "candidate_result": candidate_result.to_dict(),
        "candidate_quality": _chain_quality(candidate_result, candidate_trajectory, quality),
        "parent_result": parent_result.to_dict(),
        "parent_quality": _chain_quality(parent_result, parent_trajectory, quality),
        "exact_replay": bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and candidate_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "measured_arrival_confirmed": bool(np.max(incoming) >= 5 and np.any(bridge)),
        "teacher_active_fraction": float(
            np.mean(candidate_trajectory["shooter_loft_teacher_active"])
        ),
        "target_actor_active_fraction": float(
            np.mean(candidate_trajectory["shooter_target_velocity_contact_actor_active"])
        ),
        "candidate_artifact": candidate_artifact,
        "replay_artifact": replay_artifact,
        "parent_artifact": parent_artifact,
    }


def validate_target_contact_retention_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    claimed = report.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(report)
            or report.get("schema_version") != "rosclaw.growth.target_contact_retention.v1"
            or report.get("status") != "PASS_TARGET_CONTACT_RETENTION"
            or report.get("sealed") is not True
            or report.get("promotion_eligible") is not True
            or not all(report.get("gates", {}).values())
            or report.get("implementation_hash") != _implementation_hash()
        ):
            raise ValueError("target contact retention authority is invalid")
        request = source.parent / "request.json"
        if hash_bytes(request.read_bytes()) != report["request_hash"]:
            raise ValueError("target contact retention request changed")
        for index, row in enumerate(report["rows"]):
            case_dir = source.parent / f"case-{index:03d}"
            for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
                artifact = row[key]
                artifact_path = case_dir / artifact["file"]
                if hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]:
                    raise ValueError("target contact retention trajectory changed")
    finally:
        if claimed is not None:
            report["report_hash"] = claimed
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "target_contact_plan_actor.py",
        Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("target contact retention output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "default_target_contact_holdouts",
    "run_target_contact_retention_exam",
    "validate_target_contact_retention_exam",
]
