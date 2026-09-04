"""Matched CPU MuJoCo exam for the joint runtime finisher plan actor."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import load_runtime_contact_target_actor
from rosclaw_soccer.growth.runtime_finish_plan_actor import load_runtime_finish_plan_actor
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.runtime_contact_target_exam import _case_kwargs


def fresh_runtime_finish_plan_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Six preregistered perturbations absent from S143--S150 training."""

    origin = (5.10, -0.16406006503921598, 0.0)
    values = (
        ("s152.finish.00", 1.20935, -0.16535, 0.10085, 0.04),
        ("s152.finish.01", 1.20945, -0.16545, 0.10095, 0.02),
        ("s152.finish.02", 1.20965, -0.16565, 0.10115, 0.04),
        ("s152.finish.03", 1.20995, -0.16595, 0.10145, 0.04),
        ("s152.finish.04", 1.21355, -0.16955, 0.10375, 0.06),
        ("s152.finish.05", 1.21405, -0.17005, 0.10405, 0.06),
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
            PlaymakerPassProbeAction(body_yaw_correction_rad=yaw),
        )
        for case_id, ball_x, ball_y, friction, yaw in values
    )


def fresh_runtime_finish_plan_holdouts_v2() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Six preregistered post-S164 perturbations reserved for the sealed exam."""

    origin = (5.10, -0.16406006503921598, 0.0)
    values = (
        ("s167.finish.00", 1.20930, -0.16530, 0.10080, 0.04),
        ("s167.finish.01", 1.20940, -0.16540, 0.10090, 0.02),
        ("s167.finish.02", 1.20955, -0.16555, 0.10105, 0.04),
        ("s167.finish.03", 1.20985, -0.16585, 0.10135, 0.04),
        ("s167.finish.04", 1.21340, -0.16940, 0.10365, 0.06),
        ("s167.finish.05", 1.21375, -0.16975, 0.10385, 0.06),
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
            PlaymakerPassProbeAction(body_yaw_correction_rad=yaw),
        )
        for case_id, ball_x, ball_y, friction, yaw in values
    )


def run_runtime_finish_plan_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    finish_plan_actor_path: Path,
    finish_plan_training_report_path: Path,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    sealed: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    active_cases = cases or fresh_runtime_finish_plan_holdouts()
    if len(active_cases) != 6 or len({context.context_hash for context, _ in active_cases}) != 6:
        raise ValueError("runtime finish plan exam requires six unique cases")
    if not 1 <= workers <= 4:
        raise ValueError("runtime finish plan exam workers must be in [1, 4]")
    actor_path = finish_plan_actor_path.expanduser().resolve()
    actor = load_runtime_finish_plan_actor(actor_path)
    training_path = finish_plan_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    training_contract_valid = bool(
        (
            training.get("status") == "PASS_RUNTIME_FINISH_PLAN_TRAINING"
            and actor.continuous_policy is None
        )
        or (
            training.get("status") == "PASS_CONTINUOUS_RUNTIME_FINISH_PLAN_CALIBRATION"
            and actor.continuous_policy is not None
            and training.get("parent_actor_hash") == actor.continuous_policy.parent_actor_hash
            and training.get("critic_training_snapshot_hash")
            == actor.continuous_policy.critic_training_snapshot_hash
        )
    )
    if (
        not training_contract_valid
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or actor.neural_contact_actor_hash != neural.actor_hash
        or actor.contact_handoff_actor_hash != handoff.actor_hash
        or actor.body_hash != qualification.body_hash
        or base.body_hash != qualification.body_hash
        or neural.body_hash != qualification.body_hash
        or actor.roster_hash != base.roster_hash
        or actor.finisher_self_model_hash != base.finisher_self_model_hash
    ):
        raise ValueError("runtime finish plan exam identity changed")
    base_handoff = handoff.decide(
        contact_policy_frame=base.required_receive_action.contact_policy_frame
    )
    if not base_handoff.accepted or base_handoff.handoff_policy_frame is None:
        raise ValueError("runtime finish plan base handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_exam_request.v1",
        "partition": "SEALED_FRESH_HOLDOUT" if sealed else "CONSUMED_DEVELOPMENT_REPLAY",
        "sealed": sealed,
        "cases": [
            {"context": asdict(context), "playmaker_action": asdict(playmaker)}
            for context, playmaker in active_cases
        ],
        "context_hashes": [context.context_hash for context, _ in active_cases],
        "finish_plan_actor_hash": actor.actor_hash,
        "finish_plan_actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "finish_plan_training_report_hash": training["report_hash"],
        "base_target_actor_hash": base.actor_hash,
        "base_target_actor_file_hash": hash_bytes(base_path.read_bytes()),
        "neural_actor_hash": neural.actor_hash,
        "neural_actor_file_hash": hash_bytes(neural_path.read_bytes()),
        "handoff_actor_hash": handoff.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            actor_path,
            base_path,
            neural_path,
            output,
            index,
            context,
            playmaker,
            base_handoff.handoff_policy_frame,
            quality,
        )
        for index, (context, playmaker) in enumerate(active_cases)
    )
    if workers == 1:
        rows = [_run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, jobs))
    metrics, gates = _derive_metrics_and_gates(rows)
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_exam.v1",
        "status": (
            "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
            if sealed and passed
            else "PASS_RUNTIME_FINISH_PLAN_DEVELOPMENT"
            if passed
            else "REJECTED_RUNTIME_FINISH_PLAN"
        ),
        "sealed": sealed,
        "promotion_eligible": bool(sealed and passed),
        "partition": request["partition"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "finish_plan_actor_hash": actor.actor_hash,
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "plan_uses_only_pre_rollout_shared_team_intent": True,
            "joint_torque_owned_only_by_frozen_neural_actor": True,
            "pose_or_ball_teleport_after_start": False,
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
    return report


def validate_runtime_finish_plan_exam(path: Path) -> dict[str, Any]:
    """Validate a finish-plan exam against current code and every trajectory."""

    source = path.expanduser().resolve()
    report = _bound_json(source)
    request_path = source.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("runtime finish plan exam request must be an object")
    rows = report.get("rows")
    cases = request.get("cases")
    context_hashes = request.get("context_hashes")
    if (
        not isinstance(rows, list)
        or len(rows) != 6
        or not isinstance(cases, list)
        or len(cases) != 6
        or not isinstance(context_hashes, list)
        or len(context_hashes) != 6
        or len(set(context_hashes)) != 6
    ):
        raise ValueError("runtime finish plan exam case partition is invalid")
    metrics, gates = _derive_metrics_and_gates(cast(list[dict[str, Any]], rows))
    passed = all(gates.values())
    sealed = request.get("sealed") is True
    expected_partition = "SEALED_FRESH_HOLDOUT" if sealed else "CONSUMED_DEVELOPMENT_REPLAY"
    expected_status = (
        "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
        if sealed and passed
        else "PASS_RUNTIME_FINISH_PLAN_DEVELOPMENT"
        if passed
        else "REJECTED_RUNTIME_FINISH_PLAN"
    )
    boundary = report.get("evidence_boundary")
    if (
        request.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam_request.v1"
        or report.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam.v1"
        or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("implementation_hash") != _implementation_hash()
        or request.get("partition") != expected_partition
        or report.get("partition") != expected_partition
        or report.get("sealed") is not sealed
        or report.get("metrics") != metrics
        or report.get("gates") != gates
        or report.get("status") != expected_status
        or report.get("promotion_eligible") != bool(sealed and passed)
        or report.get("finish_plan_actor_hash") != request.get("finish_plan_actor_hash")
        or report.get("base_target_actor_hash") != request.get("base_target_actor_hash")
        or report.get("neural_actor_hash") != request.get("neural_actor_hash")
        or request.get("physics_authority") != "CPU_MUJOCO"
        or request.get("activation_ceiling") != "SIM_ONLY"
        or request.get("hardware_command_sent") is not False
        or not isinstance(boundary, dict)
        or boundary.get("whole_body_g1_count") != 3
        or boundary.get("one_shared_solver_and_ball") is not True
        or boundary.get("plan_uses_only_pre_rollout_shared_team_intent") is not True
        or boundary.get("joint_torque_owned_only_by_frozen_neural_actor") is not True
        or boundary.get("pose_or_ball_teleport_after_start") is not False
        or boundary.get("physics_authority") != "CPU_MUJOCO"
        or boundary.get("activation_ceiling") != "SIM_ONLY"
        or boundary.get("pixels_used_for_scoring") is not False
        or boundary.get("hardware_command_sent") is not False
    ):
        raise ValueError("runtime finish plan exam authority is invalid")
    for index, (case, row, context_hash) in enumerate(
        zip(cases, rows, context_hashes, strict=True)
    ):
        if (
            not isinstance(case, dict)
            or not isinstance(row, dict)
            or row.get("context") != case.get("context")
            or row.get("playmaker_action") != case.get("playmaker_action")
            or row.get("context_hash") != context_hash
            or not isinstance(case.get("context"), dict)
            or CausalTransitionContext(**case["context"]).context_hash != context_hash
            or not isinstance(case.get("playmaker_action"), dict)
        ):
            raise ValueError("runtime finish plan exam row binding changed")
        PlaymakerPassProbeAction(**case["playmaker_action"])
        case_dir = source.parent / f"case-{index:03d}"
        artifacts = {
            key: _validate_trajectory_artifact(case_dir, row.get(key))
            for key in ("candidate_artifact", "replay_artifact", "base_artifact")
        }
        exact_replay = artifacts["candidate_artifact"] == artifacts["replay_artifact"]
        if row.get("exact_replay") is not exact_replay:
            raise ValueError("runtime finish plan replay semantics changed")
    return report


def _validate_trajectory_artifact(case_dir: Path, artifact: Any) -> str:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("file"), str):
        raise ValueError("runtime finish plan trajectory contract is invalid")
    trajectory_path = (case_dir / artifact["file"]).resolve()
    if case_dir.resolve() not in trajectory_path.parents or not trajectory_path.is_file():
        raise ValueError("runtime finish plan trajectory path is invalid")
    if hash_bytes(trajectory_path.read_bytes()) != artifact.get("file_hash"):
        raise ValueError("runtime finish plan trajectory file changed")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    digest = trajectory_digest(trajectory)
    if digest != artifact.get("trajectory_digest"):
        raise ValueError("runtime finish plan trajectory semantics changed")
    return digest


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
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        actor_path,
        base_path,
        neural_path,
        output,
        index,
        context,
        playmaker,
        base_handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    base = load_runtime_contact_target_actor(base_path)
    common = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker,
        contact_actor_path=neural_path,
        target_actor=base,
        target_actor_path=base_path,
        target_velocity=None,
        handoff_frame=base_handoff_frame,
    )
    candidate = dict(common)
    candidate.pop("shooter_runtime_contact_target_actor_path")
    candidate.update(
        shooter_runtime_finish_plan_actor_path=actor_path,
        shooter_neural_contact_policy_frame=None,
        shooter_causal_strike_option_config=replace(
            common["shooter_causal_strike_option_config"],
            maximum_arrival_advance_frames=30,
        ),
    )
    candidate_result, candidate_trajectory = simulate_shared_world(asset_root, **candidate)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate)
    base_result, base_trajectory = simulate_shared_world(asset_root, **common)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    candidate_artifact = _save_trajectory(case_dir / "candidate-primary.npz", candidate_trajectory)
    replay_artifact = _save_trajectory(case_dir / "candidate-replay.npz", replay_trajectory)
    base_artifact = _save_trajectory(case_dir / "base-target-actor.npz", base_trajectory)
    candidate_quality = strict_intended_contact_quality(
        result=candidate_result,
        trajectory=candidate_trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    base_quality = strict_intended_contact_quality(
        result=base_result,
        trajectory=base_trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    decision_time = candidate_result.shooter_runtime_finish_plan_time_sec
    contact_time = candidate_result.shot_contact_time_sec
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "playmaker_action": asdict(playmaker),
        "candidate": {
            "result": candidate_result.to_dict(),
            "quality": candidate_quality,
        },
        "base": {"result": base_result.to_dict(), "quality": base_quality},
        "decision_observed": candidate_result.shooter_runtime_finish_plan_decided,
        "decision_precedes_contact": bool(
            decision_time is not None and (contact_time is None or decision_time < contact_time)
        ),
        "exact_replay": bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and candidate_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "neural_actor_active": bool(
            np.any(candidate_trajectory["shooter_neural_contact_actor_active"])
        ),
        "teacher_active": bool(np.any(candidate_trajectory["shooter_loft_teacher_active"])),
        "scripted_contact_active": bool(
            np.any(candidate_trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "candidate_artifact": candidate_artifact,
        "replay_artifact": replay_artifact,
        "base_artifact": base_artifact,
    }


def _derive_metrics_and_gates(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate_success = sum(
        bool(row["candidate"]["quality"]["strict_chain_passed"]) for row in rows
    )
    base_success = sum(bool(row["base"]["quality"]["strict_chain_passed"]) for row in rows)
    accepted_rows = [
        row for row in rows if row["candidate"]["result"]["shooter_runtime_finish_plan_accepted"]
    ]
    selected_plans = {
        (
            row["candidate"]["result"]["shooter_runtime_receive_stance_offset_y_m"],
            tuple(row["candidate"]["result"]["shooter_runtime_contact_target_velocity_xyz_mps"]),
        )
        for row in accepted_rows
    }
    metrics = {
        "case_count": len(rows),
        "accepted_count": len(accepted_rows),
        "candidate_strict_success_count": candidate_success,
        "base_strict_success_count": base_success,
        "strict_success_gain": candidate_success - base_success,
        "candidate_safe_count": sum(bool(row["candidate"]["quality"]["safe"]) for row in rows),
        "base_safe_count": sum(bool(row["base"]["quality"]["safe"]) for row in rows),
        "candidate_goal_count": sum(
            bool(row["candidate"]["result"]["goal_crossed"]) for row in rows
        ),
        "candidate_save_count": sum(
            bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in rows
        ),
        "selected_plan_count": len(selected_plans),
        "exact_replay_count": sum(bool(row["exact_replay"]) for row in rows),
    }
    gates = {
        "decision_observed_all": all(row["decision_observed"] for row in rows),
        "decision_precedes_any_contact_all": all(row["decision_precedes_contact"] for row in rows),
        "candidate_safe_six_of_six": metrics["candidate_safe_count"] == 6,
        "exact_replay_six_of_six": metrics["exact_replay_count"] == 6,
        "candidate_strict_at_least_three": candidate_success >= 3,
        "strict_gain_at_least_two": candidate_success - base_success >= 2,
        "both_goal_and_save": (
            metrics["candidate_goal_count"] >= 1 and metrics["candidate_save_count"] >= 1
        ),
        "at_least_two_plans_selected": len(selected_plans) >= 2,
        "neural_actor_realized": all(
            row["neural_actor_active"]
            for row in rows
            if row["candidate"]["result"]["shooter_runtime_finish_plan_accepted"]
        ),
        "teacher_and_scripted_contact_absent": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
    }
    return metrics, gates


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_finish_plan_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime finish plan exam output must be new and external")
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
    "fresh_runtime_finish_plan_holdouts",
    "fresh_runtime_finish_plan_holdouts_v2",
    "run_runtime_finish_plan_exam",
    "validate_runtime_finish_plan_exam",
]
