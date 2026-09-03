"""Fresh matched CPU MuJoCo exam for the role-bound runtime RECEIVE actor."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_receive_actor import load_runtime_receive_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
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


def default_runtime_receive_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Return six preregistered contexts that were absent from S131-S133."""

    origin = (5.10, -0.16406006503921598, 0.0)
    playmaker_a = PlaymakerPassProbeAction(body_yaw_correction_rad=0.04)
    playmaker_b = PlaymakerPassProbeAction(
        body_yaw_correction_rad=0.04,
        stance_correction_x_m=-0.02,
    )
    return (
        (
            CausalTransitionContext(
                "s134.holdout.00", origin, -0.0875, 1.3005, (1.2085, -0.1645), 0.80, 0.0995
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s134.holdout.01", origin, -0.0875, 1.3005, (1.2095, -0.1655), 0.80, 0.1015
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s134.holdout.02", origin, -0.0875, 1.3005, (1.2100, -0.1660), 0.80, 0.1030
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s134.holdout.03", origin, -0.0875, 1.3005, (1.2125, -0.1685), 0.80, 0.1028
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s134.holdout.04", origin, -0.0875, 1.3005, (1.2145, -0.1705), 0.80, 0.1045
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s134.holdout.05", origin, -0.0875, 1.3005, (1.2150, -0.1710), 0.80, 0.1055
            ),
            playmaker_b,
        ),
    )


def default_runtime_receive_v2_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Fresh local-retention suite around the three recovered S135 clusters."""

    origin = (5.10, -0.16406006503921598, 0.0)
    playmaker_a = PlaymakerPassProbeAction(body_yaw_correction_rad=0.04)
    playmaker_b = PlaymakerPassProbeAction(
        body_yaw_correction_rad=0.04,
        stance_correction_x_m=-0.02,
    )
    return (
        (
            CausalTransitionContext(
                "s135.holdout.v2.00",
                origin,
                -0.0875,
                1.3005,
                (1.2098, -0.1658),
                0.80,
                0.1027,
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s135.holdout.v2.01",
                origin,
                -0.0875,
                1.3005,
                (1.2102, -0.1662),
                0.80,
                0.1033,
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s135.holdout.v2.02",
                origin,
                -0.0875,
                1.3005,
                (1.2122, -0.1682),
                0.80,
                0.1025,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s135.holdout.v2.03",
                origin,
                -0.0875,
                1.3005,
                (1.2128, -0.1688),
                0.80,
                0.1031,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s135.holdout.v2.04",
                origin,
                -0.0875,
                1.3005,
                (1.2148, -0.1708),
                0.80,
                0.1052,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s135.holdout.v2.05",
                origin,
                -0.0875,
                1.3005,
                (1.2152, -0.1712),
                0.80,
                0.1058,
            ),
            playmaker_b,
        ),
    )


def default_runtime_receive_v3_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Fresh sensor-resolution holdout around S136's recovered clusters."""

    origin = (5.10, -0.16406006503921598, 0.0)
    playmaker_a = PlaymakerPassProbeAction(body_yaw_correction_rad=0.04)
    playmaker_b = PlaymakerPassProbeAction(
        body_yaw_correction_rad=0.04,
        stance_correction_x_m=-0.02,
    )
    return (
        (
            CausalTransitionContext(
                "s136.holdout.v3.00",
                origin,
                -0.0875,
                1.3005,
                (1.2096, -0.1656),
                0.80,
                0.1026,
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s136.holdout.v3.01",
                origin,
                -0.0875,
                1.3005,
                (1.2100, -0.1660),
                0.80,
                0.1028,
            ),
            playmaker_a,
        ),
        (
            CausalTransitionContext(
                "s136.holdout.v3.02",
                origin,
                -0.0875,
                1.3005,
                (1.2126, -0.1686),
                0.80,
                0.1029,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s136.holdout.v3.03",
                origin,
                -0.0875,
                1.3005,
                (1.2130, -0.1690),
                0.80,
                0.1033,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s136.holdout.v3.04",
                origin,
                -0.0875,
                1.3005,
                (1.2146, -0.1706),
                0.80,
                0.1050,
            ),
            playmaker_b,
        ),
        (
            CausalTransitionContext(
                "s136.holdout.v3.05",
                origin,
                -0.0875,
                1.3005,
                (1.2150, -0.1710),
                0.80,
                0.1054,
            ),
            playmaker_b,
        ),
    )


def run_runtime_receive_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    actor_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    sealed: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    active_cases = cases or default_runtime_receive_holdouts()
    if len(active_cases) != 6 or len({case.context_hash for case, _ in active_cases}) != 6:
        raise ValueError("runtime RECEIVE exam requires six unique contexts")
    if not 1 <= workers <= 4:
        raise ValueError("runtime RECEIVE exam workers must be in [1, 4]")
    receive_path = actor_path.expanduser().resolve()
    receive_actor = load_runtime_receive_actor(receive_path)
    neural_path = finisher_actor_path.expanduser().resolve()
    neural_actor = load_g1_neural_contact_actor(neural_path)
    teacher = _bound_json(teacher_discovery_report_path.expanduser().resolve())
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        receive_actor.body_hash != qualification.body_hash
        or receive_actor.kick_prior_hash != qualification.kick_prior_hash
        or neural_actor.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or receive_actor.agent_id != "red.finisher"
        or receive_actor.primary_role != "finisher"
        or receive_actor.tactical_intent != "receive"
        or receive_actor.owned_skill != "first_touch"
    ):
        raise ValueError("runtime RECEIVE exam identity changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("runtime RECEIVE exam needs one frozen contact target")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    contact_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=contact_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime RECEIVE exam handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_exam_request.v1",
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "sealed": sealed,
        "cases": [
            {"context": asdict(context), "playmaker_action": asdict(playmaker_action)}
            for context, playmaker_action in active_cases
        ],
        "context_hashes": [context.context_hash for context, _ in active_cases],
        "actor_hash": receive_actor.actor_hash,
        "actor_file_hash": hash_bytes(receive_path.read_bytes()),
        "finisher_actor_hash": neural_actor.actor_hash,
        "finisher_actor_file_hash": hash_bytes(neural_path.read_bytes()),
        "handoff_actor_hash": handoff.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "contact_action_hash": contact_action.action_hash,
        "roster_hash": receive_actor.roster_hash,
        "finisher_self_model_hash": receive_actor.finisher_self_model_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
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
            receive_path,
            neural_path,
            output,
            index,
            context,
            playmaker_action,
            contact_action,
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
    metrics, gates = _derive_metrics_and_gates(rows, quality)
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_exam.v1",
        "status": (
            "PASS_RUNTIME_RECEIVE_RETENTION"
            if sealed and passed
            else "PASS_RUNTIME_RECEIVE_DEVELOPMENT"
            if passed
            else "REJECTED_RUNTIME_RECEIVE"
        ),
        "sealed": sealed,
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "promotion_eligible": sealed and passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "actor_hash": receive_actor.actor_hash,
        "roster_hash": receive_actor.roster_hash,
        "finisher_self_model_hash": receive_actor.finisher_self_model_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "world_reset_after_pass_or_shot": False,
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "receive_decision_uses_only_post_pass_measured_state": True,
            "receive_law_updates_phase_from_live_eta_every_20ms": True,
            "receive_actor_pose_joint_torque_or_ball_authority": False,
            "frozen_playmaker_and_goalkeeper": True,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
    return report


def validate_runtime_receive_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _read_object(source)
    claimed = report.pop("report_hash", None)
    if claimed != hash_json(report):
        raise ValueError("runtime RECEIVE exam report integrity changed")
    report["report_hash"] = claimed
    request_path = source.parent / "request.json"
    request = _read_object(request_path)
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("runtime RECEIVE exam rows are invalid")
    quality = CausalTransitionGrowthConfig(**request["quality_config"])
    metrics, gates = _derive_metrics_and_gates(rows, quality)
    expected = (
        "PASS_RUNTIME_RECEIVE_RETENTION"
        if report.get("sealed") is True and all(gates.values())
        else "PASS_RUNTIME_RECEIVE_DEVELOPMENT"
        if all(gates.values())
        else "REJECTED_RUNTIME_RECEIVE"
    )
    boundary = report.get("evidence_boundary")
    if (
        hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("implementation_hash") != _implementation_hash()
        or report.get("metrics") != metrics
        or report.get("gates") != gates
        or report.get("status") != expected
        or report.get("promotion_eligible")
        != (report.get("sealed") is True and all(gates.values()))
        or not isinstance(boundary, dict)
        or boundary.get("physics_authority") != "CPU_MUJOCO"
        or boundary.get("activation_ceiling") != "SIM_ONLY"
        or boundary.get("receive_actor_pose_joint_torque_or_ball_authority") is not False
        or boundary.get("hardware_command_sent") is not False
        or boundary.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("runtime RECEIVE exam authority is invalid")
    for index, row in enumerate(rows):
        case_dir = source.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = row[key]
            artifact_path = case_dir / artifact["file"]
            if (
                not artifact_path.is_file()
                or hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]
            ):
                raise ValueError("runtime RECEIVE trajectory binding changed")
    return report


def _run_case(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
        TargetContactPlanAction,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        receive_actor_path,
        neural_actor_path,
        output,
        index,
        context,
        playmaker_action,
        contact_action,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_s95_dir)
    candidate_kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker_action,
        neural_actor_path=neural_actor_path,
        contact_action=contact_action,
        handoff_frame=handoff_frame,
    )
    candidate_kwargs["shooter_runtime_receive_actor_path"] = receive_actor_path
    parent_kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker_action,
        neural_actor_path=neural_actor_path,
        contact_action=contact_action,
        handoff_frame=handoff_frame,
    )
    candidate_result, candidate_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    candidate_artifact = _save_trajectory(case_dir / "candidate-primary.npz", candidate_trajectory)
    replay_artifact = _save_trajectory(case_dir / "candidate-replay.npz", replay_trajectory)
    parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trajectory)
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
    decision = candidate_result.shooter_runtime_receive_decided
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
        "runtime_decision_observed": decision,
        "candidate_actor_active": bool(
            np.any(candidate_trajectory["shooter_neural_contact_actor_active"])
        ),
        "candidate_teacher_active": bool(
            np.any(candidate_trajectory["shooter_loft_teacher_active"])
        ),
        "candidate_scripted_contact_active": bool(
            np.any(candidate_trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "candidate_phase_correction_count": int(
            np.count_nonzero(candidate_trajectory["shooter_phase_correction"])
        ),
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
    neural_actor_path: Path,
    contact_action: TargetContactPlanAction,
    handoff_frame: int,
) -> dict[str, Any]:
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
            "stance_offset_x": contact_action.stance_offset_x_m,
            "stance_offset_y": contact_action.stance_offset_y_m,
            "foot_yaw_offset": contact_action.foot_yaw_offset_rad,
            "foot_pitch_offset": contact_action.foot_pitch_offset_rad,
        },
        shooter_causal_strike_option_config=G1CausalStrikeOptionConfig(
            maximum_arrival_advance_frames=contact_action.maximum_arrival_advance_frames
        ),
        shooter_neural_contact_actor_path=neural_actor_path,
        shooter_neural_contact_policy_frame=contact_action.contact_policy_frame,
        shooter_neural_contact_target_velocity_xyz_mps=(
            contact_action.target_foot_velocity_xyz_mps
        ),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    return kwargs


def _derive_metrics_and_gates(
    rows: list[dict[str, Any]], quality: CausalTransitionGrowthConfig
) -> tuple[dict[str, Any], dict[str, bool]]:
    count = len(rows)
    candidate_success = sum(bool(row["candidate_chain_passed"]) for row in rows)
    parent_success = sum(bool(row["parent_chain_passed"]) for row in rows)
    accepted = sum(
        bool(row["candidate"]["result"]["shooter_runtime_receive_accepted"]) for row in rows
    )
    goals = sum(bool(row["candidate"]["result"]["goal_crossed"]) for row in rows)
    saves = sum(bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in rows)
    metrics = {
        "case_count": count,
        "accepted_count": accepted,
        "fallback_count": count - accepted,
        "candidate_chain_success_count": candidate_success,
        "parent_chain_success_count": parent_success,
        "candidate_chain_success_rate": candidate_success / count,
        "parent_chain_success_rate": parent_success / count,
        "chain_success_gain": candidate_success - parent_success,
        "candidate_safe_rate": sum(bool(row["candidate_safe"]) for row in rows) / count,
        "parent_safe_rate": sum(bool(row["parent_safe"]) for row in rows) / count,
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in rows) / count,
        "candidate_goal_count": goals,
        "candidate_save_count": saves,
        "mean_candidate_shot_speed_mps": float(
            np.mean([row["candidate"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "mean_parent_shot_speed_mps": float(
            np.mean([row["parent"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
    }
    gates = {
        "runtime_decision_every_case": all(row["runtime_decision_observed"] for row in rows),
        "actor_accepts_supported_majority": accepted >= 4,
        "candidate_safe_rate": metrics["candidate_safe_rate"] == 1.0,
        "parent_safe_rate": metrics["parent_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "candidate_success_rate": candidate_success / count + 1.0e-12
        >= quality.minimum_actor_success_rate,
        "candidate_not_worse_than_parent": candidate_success >= parent_success,
        "candidate_has_measured_gain": candidate_success
        >= parent_success + quality.minimum_success_gain_cases,
        "both_goal_and_save_outcomes": goals >= 1 and saves >= 1,
        "neural_actor_realized": all(
            (not row["candidate"]["result"]["shooter_runtime_receive_accepted"])
            or row["candidate_actor_active"]
            for row in rows
        ),
        "no_teacher_or_scripted_contact": all(
            not row["candidate_teacher_active"] and not row["candidate_scripted_contact_active"]
            for row in rows
        ),
        "continuous_phase_feedback_exercised": all(
            (not row["candidate"]["result"]["shooter_runtime_receive_accepted"])
            or row["candidate_phase_correction_count"] > 0
            for row in rows
        ),
        "fallbacks_fail_closed": all(
            row["candidate_safe"]
            and row["candidate"]["result"]["shooter_causal_strike_option_final_phase"] == "ABORTED"
            for row in rows
            if not row["candidate"]["result"]["shooter_runtime_receive_accepted"]
        ),
    }
    return metrics, gates


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_receive_actor.py",
        Path(__file__).parents[1] / "growth" / "causal_strike_option.py",
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "growth" / "role_self_model.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parent / "runtime_receive_growth.py",
        Path(__file__).parent / "runtime_receive_discovery.py",
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
        raise ValueError("runtime RECEIVE exam output must be new and external")
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
    "default_runtime_receive_holdouts",
    "default_runtime_receive_v2_holdouts",
    "default_runtime_receive_v3_holdouts",
    "run_runtime_receive_exam",
    "validate_runtime_receive_exam",
]
