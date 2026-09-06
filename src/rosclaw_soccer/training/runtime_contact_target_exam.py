"""Fresh matched CPU MuJoCo exam for perceptive contact-target selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    G1RuntimeContactTargetActor,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction


def fresh_runtime_contact_target_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Six preregistered conditions absent from S137--S141 target discovery."""

    origin = (5.10, -0.16406006503921598, 0.0)
    playmaker_a = PlaymakerPassProbeAction(body_yaw_correction_rad=0.04)
    playmaker_b = PlaymakerPassProbeAction(
        body_yaw_correction_rad=0.04,
        stance_correction_x_m=-0.02,
    )
    values = (
        ("s142.holdout.00", 1.2092, -0.1652, 0.1007, playmaker_a),
        ("s142.holdout.01", 1.2098, -0.1658, 0.1013, playmaker_a),
        ("s142.holdout.02", 1.2106, -0.1666, 0.1020, playmaker_a),
        ("s142.holdout.03", 1.2124, -0.1684, 0.1026, playmaker_b),
        ("s142.holdout.04", 1.2132, -0.1692, 0.1035, playmaker_b),
        ("s142.holdout.05", 1.2138, -0.1698, 0.1039, playmaker_b),
    )
    return tuple(
        (
            CausalTransitionContext(
                case_id,
                origin,
                -0.0875,
                1.3005,
                (ball_x, ball_y),
                0.80,
                friction,
            ),
            playmaker,
        )
        for case_id, ball_x, ball_y, friction, playmaker in values
    )


def run_runtime_contact_target_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    target_actor_path: Path,
    target_training_report_path: Path,
    neural_actor_path: Path,
    neural_training_report_path: Path,
    parent_neural_actor_path: Path,
    parent_training_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    sealed: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Compare contextual target selection with the frozen single-target parent."""

    active_cases = cases or fresh_runtime_contact_target_holdouts()
    if len(active_cases) != 6 or len({context.context_hash for context, _ in active_cases}) != 6:
        raise ValueError("runtime contact target exam requires six unique contexts")
    if not 1 <= workers <= 4:
        raise ValueError("runtime contact target exam workers must be in [1, 4]")
    target_path = target_actor_path.expanduser().resolve()
    target_actor = load_runtime_contact_target_actor(target_path)
    target_training_path = target_training_report_path.expanduser().resolve()
    target_training = _bound_json(target_training_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural_actor = load_g1_neural_contact_actor(neural_path)
    neural_training_path = neural_training_report_path.expanduser().resolve()
    neural_training = _bound_json(neural_training_path)
    parent_path = parent_neural_actor_path.expanduser().resolve()
    parent_actor = load_g1_neural_contact_actor(parent_path)
    parent_training_path = parent_training_report_path.expanduser().resolve()
    parent_training = _bound_json(parent_training_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        target_training.get("status") != "PASS_RUNTIME_CONTACT_TARGET_TRAINING"
        or target_training.get("actor_hash") != target_actor.actor_hash
        or target_training.get("actor_file_hash") != hash_bytes(target_path.read_bytes())
        or neural_training.get("status") != "PASS_NEURAL_CONTACT_DISTILLATION"
        or neural_training.get("actor_hash") != neural_actor.actor_hash
        or neural_training.get("actor_file_hash") != hash_bytes(neural_path.read_bytes())
        or parent_training.get("status") != "PASS_NEURAL_CONTACT_DISTILLATION"
        or parent_training.get("actor_hash") != parent_actor.actor_hash
        or parent_training.get("actor_file_hash") != hash_bytes(parent_path.read_bytes())
        or target_actor.neural_contact_actor_hash != neural_actor.actor_hash
        or target_actor.body_hash != qualification.body_hash
        or neural_actor.body_hash != qualification.body_hash
        or parent_actor.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or not parent_actor.target_supported((9.0, -3.0, -1.0))
        or target_actor.agent_id != "red.finisher"
        or target_actor.owned_skill != "contact_target_selection"
    ):
        raise ValueError("runtime contact target exam identity changed")
    required = target_actor.required_receive_action
    handoff_decision = handoff.decide(contact_policy_frame=required.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime contact target exam handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_contact_target_exam_request.v1",
        "partition": "SEALED_FRESH_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "sealed": sealed,
        "cases": [
            {"context": asdict(context), "playmaker_action": asdict(playmaker_action)}
            for context, playmaker_action in active_cases
        ],
        "context_hashes": [context.context_hash for context, _ in active_cases],
        "target_actor_hash": target_actor.actor_hash,
        "target_actor_file_hash": hash_bytes(target_path.read_bytes()),
        "target_training_report_hash": target_training["report_hash"],
        "neural_actor_hash": neural_actor.actor_hash,
        "neural_actor_file_hash": hash_bytes(neural_path.read_bytes()),
        "neural_training_report_hash": neural_training["report_hash"],
        "parent_neural_actor_hash": parent_actor.actor_hash,
        "parent_neural_actor_file_hash": hash_bytes(parent_path.read_bytes()),
        "parent_training_report_hash": parent_training["report_hash"],
        "parent_target_velocity_xyz_mps": [9.0, -3.0, -1.0],
        "required_receive_action": asdict(required),
        "handoff_actor_hash": handoff.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            target_path,
            neural_path,
            parent_path,
            output,
            index,
            context,
            playmaker_action,
            target_actor,
            handoff_decision.handoff_policy_frame,
            quality,
        )
        for index, (context, playmaker_action) in enumerate(active_cases)
    )
    if workers == 1:
        rows = [_run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, jobs))
    metrics, gates = _derive_metrics_and_gates(rows)
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_contact_target_exam.v1",
        "status": (
            "PASS_RUNTIME_CONTACT_TARGET_FRESH_HOLDOUT"
            if sealed and passed
            else "PASS_RUNTIME_CONTACT_TARGET_DEVELOPMENT"
            if passed
            else "REJECTED_RUNTIME_CONTACT_TARGET"
        ),
        "sealed": sealed,
        "partition": "SEALED_FRESH_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "promotion_eligible": bool(sealed and passed),
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "target_actor_hash": target_actor.actor_hash,
        "neural_actor_hash": neural_actor.actor_hash,
        "parent_neural_actor_hash": parent_actor.actor_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "target_decision_uses_only_pre_action_measured_state": True,
            "target_actor_task_space_only": True,
            "neural_actor_is_sole_contact_torque_residual": True,
            "frozen_playmaker_and_goalkeeper": True,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
    return report


def validate_runtime_contact_target_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _bound_json(source)
    request_path = source.parent / "request.json"
    request = _read_object(request_path)
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("runtime contact target exam rows are invalid")
    metrics, gates = _derive_metrics_and_gates(cast(list[dict[str, Any]], rows))
    passed = all(gates.values())
    expected = (
        "PASS_RUNTIME_CONTACT_TARGET_FRESH_HOLDOUT"
        if report.get("sealed") is True and passed
        else "PASS_RUNTIME_CONTACT_TARGET_DEVELOPMENT"
        if passed
        else "REJECTED_RUNTIME_CONTACT_TARGET"
    )
    boundary = report.get("evidence_boundary")
    if (
        hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("implementation_hash") != _implementation_hash()
        or report.get("metrics") != metrics
        or report.get("gates") != gates
        or report.get("status") != expected
        or report.get("promotion_eligible") != bool(report.get("sealed") is True and passed)
        or not isinstance(boundary, dict)
        or boundary.get("physics_authority") != "CPU_MUJOCO"
        or boundary.get("target_actor_task_space_only") is not True
        or boundary.get("neural_actor_is_sole_contact_torque_residual") is not True
        or boundary.get("hardware_command_sent") is not False
        or boundary.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("runtime contact target exam authority is invalid")
    for index, row in enumerate(rows):
        case_dir = source.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = row[key]
            artifact_path = case_dir / artifact["file"]
            if hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("runtime contact target trajectory changed")
    return report


def _run_case(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
        G1RuntimeContactTargetActor,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        target_actor_path,
        neural_actor_path,
        parent_actor_path,
        output,
        index,
        context,
        playmaker_action,
        target_actor,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_s95_dir)
    candidate_kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker_action,
        contact_actor_path=neural_actor_path,
        target_actor=target_actor,
        target_actor_path=target_actor_path,
        target_velocity=None,
        handoff_frame=handoff_frame,
    )
    parent_kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker_action,
        contact_actor_path=parent_actor_path,
        target_actor=target_actor,
        target_actor_path=None,
        target_velocity=(9.0, -3.0, -1.0),
        handoff_frame=handoff_frame,
    )
    candidate_result, candidate_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    candidate_artifact = _save_trajectory(case_dir / "candidate-primary.npz", candidate_trajectory)
    replay_artifact = _save_trajectory(case_dir / "candidate-replay.npz", replay_trajectory)
    parent_artifact = _save_trajectory(case_dir / "parent-specialist.npz", parent_trajectory)
    candidate_quality = strict_intended_contact_quality(
        result=candidate_result,
        trajectory=candidate_trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    parent_quality = strict_intended_contact_quality(
        result=parent_result,
        trajectory=parent_trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    target_trace = np.asarray(
        candidate_trajectory["shooter_runtime_contact_target_velocity_xyz_mps"],
        dtype=np.float64,
    )
    selected_rows = target_trace[np.linalg.norm(target_trace, axis=1) > 0.0]
    selected_targets = {tuple(float(value) for value in row) for row in selected_rows}
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "playmaker_action": asdict(playmaker_action),
        "candidate": {"result": candidate_result.to_dict(), "quality": candidate_quality},
        "parent": {"result": parent_result.to_dict(), "quality": parent_quality},
        "candidate_chain_passed": candidate_quality["strict_chain_passed"],
        "parent_chain_passed": parent_quality["strict_chain_passed"],
        "candidate_safe": candidate_quality["safe"],
        "parent_safe": parent_quality["safe"],
        "exact_replay": bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and candidate_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "runtime_target_decision_observed": (
            candidate_result.shooter_runtime_contact_target_decided
        ),
        "runtime_target_decision_precedes_contact": bool(
            candidate_result.shooter_runtime_contact_target_time_sec is not None
            and candidate_result.shot_contact_time_sec is not None
            and candidate_result.shooter_runtime_contact_target_time_sec
            < candidate_result.shot_contact_time_sec
        ),
        "selected_target_latched": len(selected_targets) <= 1,
        "candidate_actor_active": bool(
            np.any(candidate_trajectory["shooter_neural_contact_actor_active"])
        ),
        "candidate_teacher_active": bool(
            np.any(candidate_trajectory["shooter_loft_teacher_active"])
        ),
        "candidate_scripted_contact_active": bool(
            np.any(candidate_trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "failure_route": _failure_route(candidate_quality, candidate_result.to_dict()),
        "candidate_artifact": candidate_artifact,
        "replay_artifact": replay_artifact,
        "parent_artifact": parent_artifact,
    }


def _case_kwargs(
    *,
    lead: Any,
    quality: CausalTransitionGrowthConfig,
    context: CausalTransitionContext,
    playmaker_action: PlaymakerPassProbeAction,
    contact_actor_path: Path,
    target_actor: G1RuntimeContactTargetActor,
    target_actor_path: Path | None,
    target_velocity: tuple[float, float, float] | None,
    handoff_frame: int,
) -> dict[str, Any]:
    required = target_actor.required_receive_action
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    base = cast(dict[str, float], kwargs["passer_parameter_overrides"])
    kwargs["passer_parameter_overrides"] = {
        **base,
        "stance_offset_x": base["stance_offset_x"] + playmaker_action.stance_correction_x_m,
        "stance_offset_y": base["stance_offset_y"] + playmaker_action.stance_correction_y_m,
        "swing_speed_scale": playmaker_action.swing_speed_scale,
    }
    base_yaw = float(kwargs["passer_yaw_rad"])
    kwargs["passer_yaw_rad"] = math.atan2(
        math.sin(base_yaw + playmaker_action.body_yaw_correction_rad),
        math.cos(base_yaw + playmaker_action.body_yaw_correction_rad),
    )
    kwargs.update(
        shooter_parameter_overrides={
            "stance_offset_x": required.stance_offset_x_m,
            "stance_offset_y": required.stance_offset_y_m,
            "foot_yaw_offset": required.foot_yaw_offset_rad,
            "foot_pitch_offset": required.foot_pitch_offset_rad,
        },
        shooter_causal_strike_option_config=replace(
            G1CausalStrikeOptionConfig(),
            maximum_arrival_advance_frames=required.maximum_arrival_advance_frames,
            arrival_alignment_tolerance_sec=required.arrival_alignment_tolerance_sec,
        ),
        shooter_runtime_contact_target_actor_path=target_actor_path,
        shooter_neural_contact_actor_path=contact_actor_path,
        shooter_neural_contact_policy_frame=required.contact_policy_frame,
        shooter_neural_contact_target_velocity_xyz_mps=target_velocity,
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    return kwargs


def _failure_route(quality: dict[str, Any], result: dict[str, Any]) -> str:
    if not quality["safe"]:
        return "POST_CONTACT_STABILITY"
    if not result["pass_contact_observed"] or not result["shot_contact_observed"]:
        return "PHASE_OR_CONTACT_ACQUISITION"
    if not quality["intended_foot_contact"]:
        return "INTENDED_FOOT_ALIGNMENT"
    if not quality["clear_outcome"]:
        return "SHOT_DIRECTION"
    if not quality["strict_chain_passed"]:
        return "SHOT_SPEED_OR_ORDERING"
    return "SUCCESS"


def _derive_metrics_and_gates(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    count = len(rows)
    candidate_success = sum(bool(row["candidate_chain_passed"]) for row in rows)
    parent_success = sum(bool(row["parent_chain_passed"]) for row in rows)
    accepted = sum(
        bool(row["candidate"]["result"]["shooter_runtime_contact_target_accepted"]) for row in rows
    )
    goals = sum(bool(row["candidate"]["result"]["goal_crossed"]) for row in rows)
    saves = sum(bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in rows)
    selected_targets = {
        tuple(row["candidate"]["result"]["shooter_runtime_contact_target_velocity_xyz_mps"])
        for row in rows
        if row["candidate"]["result"]["shooter_runtime_contact_target_velocity_xyz_mps"] is not None
    }
    failure_counts: dict[str, int] = {}
    for row in rows:
        route = str(row["failure_route"])
        failure_counts[route] = failure_counts.get(route, 0) + 1
    metrics = {
        "case_count": count,
        "accepted_count": accepted,
        "candidate_strict_success_count": candidate_success,
        "parent_strict_success_count": parent_success,
        "strict_success_gain": candidate_success - parent_success,
        "candidate_goal_count": goals,
        "candidate_save_count": saves,
        "selected_target_count": len(selected_targets),
        "selected_targets_xyz_mps": [list(target) for target in sorted(selected_targets)],
        "candidate_safe_count": sum(bool(row["candidate_safe"]) for row in rows),
        "parent_safe_count": sum(bool(row["parent_safe"]) for row in rows),
        "exact_replay_count": sum(bool(row["exact_replay"]) for row in rows),
        "mean_candidate_shot_speed_mps": float(
            np.mean([row["candidate"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "mean_parent_shot_speed_mps": float(
            np.mean([row["parent"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "failure_route_counts": failure_counts,
    }
    gates = {
        "runtime_target_decision_all": all(row["runtime_target_decision_observed"] for row in rows),
        "runtime_target_decision_precedes_contact_all": all(
            row["runtime_target_decision_precedes_contact"] for row in rows
        ),
        "target_latched_all": all(row["selected_target_latched"] for row in rows),
        "actor_accepts_supported_majority": accepted >= 4,
        "candidate_safe_six_of_six": metrics["candidate_safe_count"] == count,
        "parent_safe_six_of_six": metrics["parent_safe_count"] == count,
        "exact_replay_six_of_six": metrics["exact_replay_count"] == count,
        "candidate_strict_at_least_four": candidate_success >= 4,
        "candidate_gains_at_least_one": candidate_success >= parent_success + 1,
        "both_goal_and_save_outcomes": goals >= 1 and saves >= 1,
        "contextual_target_diversity": len(selected_targets) >= 2,
        "neural_actor_realized": all(
            (not row["candidate"]["result"]["shooter_runtime_contact_target_accepted"])
            or row["candidate_actor_active"]
            for row in rows
        ),
        "teacher_and_scripted_contact_absent": all(
            not row["candidate_teacher_active"] and not row["candidate_scripted_contact_active"]
            for row in rows
        ),
    }
    return metrics, gates


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_contact_target_actor.py",
        Path(__file__).parents[1] / "growth" / "runtime_receive_actor.py",
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parent / "runtime_contact_target_growth.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime contact target exam output must be new and external")
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
    "fresh_runtime_contact_target_holdouts",
    "run_runtime_contact_target_exam",
    "validate_runtime_contact_target_exam",
]
