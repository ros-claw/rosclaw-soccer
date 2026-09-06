"""Fresh sealed exam for planned-plus-reactive learned contact control."""

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
from rosclaw_soccer.growth.runtime_contact_mode_actor import (
    load_runtime_contact_mode_actor,
)
from rosclaw_soccer.growth.three_axis_contact_actor import (
    load_g1_three_axis_contact_actor,
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


def default_dual_clock_contact_holdouts() -> tuple[CausalTransitionContext, ...]:
    """S128 v4 contexts registered before their first physics rollout."""

    origin = (5.10, -0.16406006503921598, 0.0)
    return (
        CausalTransitionContext(
            "s128.holdout.v4.00", origin, -0.108, 1.338, (1.193, -0.151), 0.80, 0.0915
        ),
        CausalTransitionContext(
            "s128.holdout.v4.01", origin, -0.115, 1.344, (1.188, -0.148), 0.80, 0.0885
        ),
        CausalTransitionContext(
            "s128.holdout.v4.02", origin, -0.082, 1.304, (1.209, -0.165), 0.80, 0.1030
        ),
        CausalTransitionContext(
            "s128.holdout.v4.03", origin, -0.087, 1.300, (1.212, -0.168), 0.80, 0.1020
        ),
        CausalTransitionContext(
            "s128.holdout.v4.04", origin, 0.097, 1.218, (1.223, -0.178), 0.80, 0.1115
        ),
        CausalTransitionContext(
            "s128.holdout.v4.05", origin, 0.103, 1.212, (1.227, -0.182), 0.80, 0.1130
        ),
    )


def run_dual_clock_contact_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    planned_actor_path: Path,
    runtime_actor_path: Path,
    contact_actor_path: Path,
    frozen_parent_router_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    holdouts: tuple[CausalTransitionContext, ...] | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    contexts = holdouts or default_dual_clock_contact_holdouts()
    if len(contexts) != 6 or len({context.context_hash for context in contexts}) != 6:
        raise ValueError("dual-clock contact exam requires six unique holdouts")
    if not 1 <= workers <= 6:
        raise ValueError("dual-clock contact exam workers must be in [1, 6]")
    quality = quality_config or CausalTransitionGrowthConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    planned = load_planned_contact_mode_actor(planned_actor_path)
    runtime = load_runtime_contact_mode_actor(runtime_actor_path)
    contact = load_g1_three_axis_contact_actor(contact_actor_path)
    for actor in (planned, runtime, contact):
        if actor.body_hash != qualification.body_hash:
            raise ValueError("dual-clock actor Body identity changed")
    if planned.kick_prior_hash != qualification.kick_prior_hash or (
        runtime.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("dual-clock actor KickPrior identity changed")
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    implementation_hash = _implementation_hash()
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.dual_clock_contact_exam_request.v1",
        "partition": "SEALED_HOLDOUT",
        "sealed": True,
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "planned_actor_hash": planned.actor_hash,
        "planned_actor_file_hash": hash_bytes(planned_actor_path.read_bytes()),
        "runtime_actor_hash": runtime.actor_hash,
        "runtime_actor_file_hash": hash_bytes(runtime_actor_path.read_bytes()),
        "contact_actor_hash": contact.actor_hash,
        "contact_actor_file_hash": hash_bytes(contact_actor_path.read_bytes()),
        "frozen_parent_router_file_hash": hash_bytes(frozen_parent_router_path.read_bytes()),
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "source_s95_evidence_hash": source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": implementation_hash,
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
            planned_actor_path.expanduser().resolve(),
            runtime_actor_path.expanduser().resolve(),
            contact_actor_path.expanduser().resolve(),
            frozen_parent_router_path.expanduser().resolve(),
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
    candidate_goal = sum(bool(row["candidate_result"]["goal_crossed"]) for row in rows)
    candidate_save = sum(bool(row["candidate_result"]["goalkeeper_save_observed"]) for row in rows)
    metrics = {
        "case_count": len(rows),
        "candidate_chain_success_count": candidate_success,
        "parent_chain_success_count": parent_success,
        "chain_success_gain": candidate_success - parent_success,
        "candidate_safe_rate": sum(bool(row["candidate_quality"]["safe"]) for row in rows)
        / len(rows),
        "parent_safe_rate": sum(bool(row["parent_quality"]["safe"]) for row in rows) / len(rows),
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in rows) / len(rows),
        "planned_accept_count": sum(bool(row["planned_decision"]["accepted"]) for row in rows),
        "runtime_accept_count": sum(bool(row["runtime_decision"]["accepted"]) for row in rows),
        "candidate_goal_count": candidate_goal,
        "candidate_save_count": candidate_save,
        "mean_candidate_shot_speed_mps": float(
            np.mean([row["candidate_result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
    }
    cluster_success = tuple(
        any(bool(rows[index]["candidate_quality"]["chain_passed"]) for index in cluster)
        for cluster in ((0, 1), (2, 3), (4, 5))
    )
    gates = {
        "planned_actor_accepts_all": metrics["planned_accept_count"] == len(rows),
        "runtime_actor_accepts_all": metrics["runtime_accept_count"] == len(rows),
        "candidate_safe_rate": metrics["candidate_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "candidate_success_rate": candidate_success / len(rows) + 1.0e-12
        >= quality.minimum_actor_success_rate,
        "candidate_has_measured_gain": candidate_success
        >= parent_success + quality.minimum_success_gain_cases,
        "each_support_cluster_succeeds": all(cluster_success),
        "both_goal_and_save_outcomes": candidate_goal >= 1 and candidate_save >= 1,
        "teacher_absent": all(row["teacher_active_fraction"] == 0.0 for row in rows),
        "learned_contact_actor_executed": all(
            row["contact_actor_active_fraction"] > 0.0 for row in rows
        ),
        "deterministic_plan_runtime_commitment": all(
            row["planned_decision"] == row["replay_planned_decision"]
            and row["runtime_decision"] == row["replay_runtime_decision"]
            for row in rows
        ),
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.dual_clock_contact_exam.v1",
        "status": (
            "PASS_DUAL_CLOCK_CONTACT_RETENTION"
            if passed
            else "REJECTED_DUAL_CLOCK_CONTACT_RETENTION"
        ),
        "sealed": True,
        "partition": "SEALED_HOLDOUT",
        "promotion_eligible": passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "planned_actor_hash": planned.actor_hash,
        "runtime_actor_hash": runtime.actor_hash,
        "contact_actor_hash": contact.actor_hash,
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
            "runtime_route_uses_only_post_pass_measured_state": True,
            "teacher_enabled": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
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
        int,
        CausalTransitionContext,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        planned_actor_path,
        runtime_actor_path,
        contact_actor_path,
        frozen_parent_router_path,
        output,
        index,
        context,
        quality,
    ) = job
    planned = load_planned_contact_mode_actor(planned_actor_path)
    plan_features = planned_contact_mode_features(
        receiver_lane_m=context.receiver_lane_m,
        reception_target_x_m=context.reception_target_x_m,
        passer_ball_local_xy_m=context.passer_ball_local_xy_m,
        ball_ground_friction=context.ball_ground_friction,
    )
    plan = planned.decide(plan_features)
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    candidate_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    if plan.action is not None:
        candidate_kwargs.update(
            shooter_parameter_overrides={
                "stance_offset_x": plan.action.stance_offset_x_m,
                "stance_offset_y": plan.action.stance_offset_y_m,
                "foot_yaw_offset": plan.action.foot_yaw_offset_rad,
                "foot_pitch_offset": plan.action.foot_pitch_offset_rad,
            },
            shooter_causal_strike_option_config=G1CausalStrikeOptionConfig(),
            shooter_runtime_contact_mode_actor_path=runtime_actor_path,
            shooter_ballistic_contact_torque_config=replace(
                UpperCornerStrikePolicy().torque_config(),
                contact_policy_frame=plan.action.contact_policy_frame,
            ),
            shooter_three_axis_contact_actor_path=contact_actor_path,
            shooter_precontact_joint_guard_enabled=True,
        )
    parent_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    parent_kwargs.update(
        shooter_causal_strike_option_config=G1CausalStrikeOptionConfig(),
        shooter_runtime_strike_router_path=frozen_parent_router_path,
        shooter_ballistic_contact_torque_config=UpperCornerStrikePolicy().torque_config(),
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
    runtime = load_runtime_contact_mode_actor(runtime_actor_path)
    runtime_decision = _runtime_decision(runtime, candidate_trajectory)
    replay_runtime_decision = _runtime_decision(runtime, replay_trajectory)
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "planned_decision": asdict(plan),
        "replay_planned_decision": asdict(planned.decide(plan_features)),
        "runtime_decision": runtime_decision,
        "replay_runtime_decision": replay_runtime_decision,
        "candidate_result": candidate_result.to_dict(),
        "candidate_quality": _chain_quality(candidate_result, candidate_trajectory, quality),
        "parent_result": parent_result.to_dict(),
        "parent_quality": _chain_quality(parent_result, parent_trajectory, quality),
        "exact_replay": bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and candidate_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "teacher_active_fraction": float(
            np.mean(candidate_trajectory["shooter_loft_teacher_active"])
        ),
        "contact_actor_active_fraction": float(
            np.mean(candidate_trajectory["shooter_three_axis_contact_actor_active"])
        ),
        "candidate_artifact": candidate_artifact,
        "replay_artifact": replay_artifact,
        "parent_artifact": parent_artifact,
    }


def _runtime_decision(actor: Any, trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    decided = np.flatnonzero(trajectory["shooter_runtime_strike_route_decided"])
    if decided.size == 0:
        return {
            "accepted": False,
            "route": "NO_RUNTIME_DECISION",
            "action": None,
            "actor_hash": actor.actor_hash,
        }
    features = trajectory["shooter_runtime_strike_features"][int(decided[0])]
    return asdict(actor.decide(features))


def validate_dual_clock_contact_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    claimed = report.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(report)
            or report.get("schema_version") != "rosclaw.growth.dual_clock_contact_exam.v1"
            or report.get("status") != "PASS_DUAL_CLOCK_CONTACT_RETENTION"
            or report.get("sealed") is not True
            or report.get("promotion_eligible") is not True
            or not all(report.get("gates", {}).values())
            or report.get("implementation_hash") != _implementation_hash()
        ):
            raise ValueError("dual-clock contact exam authority is invalid")
        request = source.parent / "request.json"
        if not request.is_file() or hash_bytes(request.read_bytes()) != report["request_hash"]:
            raise ValueError("dual-clock exam request binding changed")
        for index, row in enumerate(report["rows"]):
            case_dir = source.parent / f"case-{index:03d}"
            for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
                artifact = row[key]
                artifact_path = case_dir / artifact["file"]
                if (
                    not artifact_path.is_file()
                    or hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]
                ):
                    raise ValueError("dual-clock exam trajectory binding changed")
    finally:
        if claimed is not None:
            report["report_hash"] = claimed
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "planned_contact_mode_actor.py",
        Path(__file__).parents[1] / "growth" / "runtime_contact_mode_actor.py",
        Path(__file__).parents[1] / "growth" / "three_axis_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dual-clock exam output must be new and external")
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
    "default_dual_clock_contact_holdouts",
    "run_dual_clock_contact_exam",
    "validate_dual_clock_contact_exam",
]
