"""Discover playmaker actions that enter the finisher's learned support."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.playmaker_pass_actor import load_playmaker_pass_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import load_runtime_contact_target_actor
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.runtime_contact_target_exam import _case_kwargs
from rosclaw_soccer.training.runtime_receive_growth import extract_runtime_receive_features


def default_team_compatible_pass_actions() -> tuple[PlaymakerPassProbeAction, ...]:
    actions = (
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04),
        PlaymakerPassProbeAction(
            body_yaw_correction_rad=0.04,
            stance_correction_x_m=-0.02,
        ),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.06),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, swing_speed_scale=0.84),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.06, swing_speed_scale=0.84),
    )
    if len({action.action_hash for action in actions}) != len(actions):
        raise RuntimeError("team-compatible pass actions are not unique")
    return actions


def run_team_compatible_pass_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_target_exam_path: Path,
    rejected_stance_report_path: Path,
    prior_playmaker_actor_path: Path,
    target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    actions: tuple[PlaymakerPassProbeAction, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("team-compatible pass discovery workers must be in [1, 4]")
    target_exam_path = rejected_target_exam_path.expanduser().resolve()
    target_exam = _bound_rejected_target_exam(target_exam_path)
    stance_path = rejected_stance_report_path.expanduser().resolve()
    stance = _bound_rejected_stance_report(stance_path)
    cases = tuple(_context_from_dict(row["context"]) for row in target_exam["rows"])
    active_actions = actions or default_team_compatible_pass_actions()
    if (
        len(cases) != 6
        or len({context.context_hash for context in cases}) != 6
        or len(active_actions) != 5
        or len({action.action_hash for action in active_actions}) != 5
    ):
        raise ValueError("team-compatible pass curriculum is invalid")
    prior_path = prior_playmaker_actor_path.expanduser().resolve()
    prior = load_playmaker_pass_actor(prior_path)
    target_path = target_actor_path.expanduser().resolve()
    target_actor = load_runtime_contact_target_actor(target_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        target_exam.get("target_actor_hash") != target_actor.actor_hash
        or target_exam.get("neural_actor_hash") != neural.actor_hash
        or stance.get("target_actor_hash") != target_actor.actor_hash
        or stance.get("neural_actor_hash") != neural.actor_hash
        or prior.body_hash != qualification.body_hash
        or target_actor.body_hash != qualification.body_hash
        or target_actor.neural_contact_actor_hash != neural.actor_hash
        or handoff.body_hash != qualification.body_hash
    ):
        raise ValueError("team-compatible pass discovery identity changed")
    required = target_actor.required_receive_action
    handoff_decision = handoff.decide(contact_policy_frame=required.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("team-compatible pass discovery handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.team_compatible_pass_discovery_request.v1",
        "partition": "CONSUMED_S142_TEAM_CREDIT_REPAIR",
        "plastic_agent_id": "red.playmaker",
        "plastic_skill": "support_compatible_lead_pass",
        "contexts": [asdict(context) for context in cases],
        "context_hashes": [context.context_hash for context in cases],
        "actions": [asdict(action) for action in active_actions],
        "action_hashes": [action.action_hash for action in active_actions],
        "rejected_target_exam_hash": target_exam["report_hash"],
        "rejected_target_exam_file_hash": hash_bytes(target_exam_path.read_bytes()),
        "rejected_stance_report_hash": stance["report_hash"],
        "rejected_stance_report_file_hash": hash_bytes(stance_path.read_bytes()),
        "prior_playmaker_actor_hash": prior.actor_hash,
        "prior_playmaker_actor_file_hash": hash_bytes(prior_path.read_bytes()),
        "frozen_target_actor_hash": target_actor.actor_hash,
        "frozen_target_actor_file_hash": hash_bytes(target_path.read_bytes()),
        "frozen_neural_actor_hash": neural.actor_hash,
        "frozen_neural_actor_file_hash": hash_bytes(neural_path.read_bytes()),
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "frozen_handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "implementation_hash": _implementation_hash(),
        "credit_contract": {
            "plastic_role": "red.playmaker",
            "frozen_roles": ["red.finisher", "blue.goalkeeper"],
            "arrival_support_measured_after_physical_pass": True,
            "pixels_used_for_selection": False,
        },
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            target_path,
            neural_path,
            output,
            index,
            context,
            action,
            target_actor,
            handoff_decision.handoff_policy_frame,
            quality,
        )
        for index, (context, action) in enumerate(
            (context, action) for context in cases for action in active_actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select(rows, context.context_hash) for context in cases]
    recovered = {row["context_hash"] for row in rows if row["quality"]["strict_chain_passed"]}
    gates = {
        "minimum_four_contexts_recovered": len(recovered) >= 4,
        "selected_safe_all": all(row["quality"]["safe"] for row in selected),
        "selected_target_supported_all": all(row["target_accepted"] for row in selected),
        "selected_strict_at_least_four": sum(
            bool(row["quality"]["strict_chain_passed"]) for row in selected
        )
        >= 4,
        "both_goal_and_save": any(row["result"]["goal_crossed"] for row in selected)
        and any(row["result"]["goalkeeper_save_observed"] for row in selected),
        "teacher_and_scripted_contact_absent": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.team_compatible_pass_discovery.v1",
        "status": (
            "PASS_TEAM_COMPATIBLE_PASS_DISCOVERY"
            if all(gates.values())
            else "REJECTED_TEAM_COMPATIBLE_PASS_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_S142_TEAM_CREDIT_REPAIR",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "plastic_agent_id": "red.playmaker",
        "prior_playmaker_actor_hash": prior.actor_hash,
        "frozen_target_actor_hash": target_actor.actor_hash,
        "frozen_neural_actor_hash": neural.actor_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "metrics": {
            "probe_count": len(rows),
            "safe_count": sum(bool(row["quality"]["safe"]) for row in rows),
            "target_accepted_count": sum(bool(row["target_accepted"]) for row in rows),
            "strict_success_count": sum(
                bool(row["quality"]["strict_chain_passed"]) for row in rows
            ),
            "recovered_context_count": len(recovered),
            "selected_strict_count": sum(
                bool(row["quality"]["strict_chain_passed"]) for row in selected
            ),
            "selected_mean_target_support_distance": float(
                np.mean([row["target_support_distance"] for row in selected])
            ),
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "credit_contract": request["credit_contract"],
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
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
        Path,
        Path,
        int,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
        Any,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        target_path,
        neural_path,
        output,
        index,
        context,
        action,
        target_actor,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=action,
        contact_actor_path=neural_path,
        target_actor=target_actor,
        target_actor_path=target_path,
        target_velocity=None,
        handoff_frame=handoff_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    relative = Path(f"probe-{index:03d}.npz")
    artifact = _save_trajectory(output / relative, trajectory)
    artifact["file"] = str(relative)
    quality_result = strict_intended_contact_quality(
        result=result,
        trajectory=trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    support = result.shooter_runtime_contact_target_support_distance
    return {
        "probe_index": index,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "playmaker_action": asdict(action),
        "playmaker_action_hash": action.action_hash,
        "pre_action_receive_features": list(extract_runtime_receive_features(output / relative)),
        "target_accepted": result.shooter_runtime_contact_target_accepted,
        "target_support_distance": 1.0e9 if support is None else support,
        "selected_target_velocity_xyz_mps": (
            result.shooter_runtime_contact_target_velocity_xyz_mps
        ),
        "result": result.to_dict(),
        "quality": quality_result,
        "neural_actor_active": bool(np.any(trajectory["shooter_neural_contact_actor_active"])),
        "teacher_active": bool(np.any(trajectory["shooter_loft_teacher_active"])),
        "scripted_contact_active": bool(
            np.any(trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "trajectory": artifact,
    }


def _select(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["context_hash"] == context_hash]
    return min(
        candidates,
        key=lambda row: (
            not row["quality"]["strict_chain_passed"],
            not row["quality"]["safe"],
            not row["target_accepted"],
            row["target_support_distance"],
            not row["quality"]["intended_foot_contact"],
            not row["quality"]["clear_outcome"],
            -float(row["result"]["shot_peak_ball_speed_mps"]),
            row["playmaker_action_hash"],
        ),
    )


def _bound_rejected_target_exam(path: Path) -> dict[str, Any]:
    report = _bound_json(path)
    request = path.parent / "request.json"
    if (
        report.get("schema_version") != "rosclaw_soccer.runtime_contact_target_exam.v1"
        or report.get("status") != "REJECTED_RUNTIME_CONTACT_TARGET"
        or report.get("sealed") is not True
        or report.get("promotion_eligible") is not False
        or not request.is_file()
        or hash_bytes(request.read_bytes()) != report.get("request_hash")
        or len(report.get("rows", ())) != 6
    ):
        raise ValueError("rejected target exam evidence is invalid")
    return report


def _bound_rejected_stance_report(path: Path) -> dict[str, Any]:
    report = _bound_json(path)
    request = path.parent / "request.json"
    if (
        report.get("schema_version") != "rosclaw_soccer.runtime_ready_stance_repair.v1"
        or report.get("status") != "REJECTED_RUNTIME_READY_STANCE_FOUNDATION"
        or report.get("promotion_eligible") is not False
        or not request.is_file()
        or hash_bytes(request.read_bytes()) != report.get("request_hash")
        or len(report.get("rows", ())) != 20
    ):
        raise ValueError("rejected stance evidence is invalid")
    for row in report["rows"]:
        artifact = row["trajectory"]
        if hash_bytes((path.parent / artifact["file"]).read_bytes()) != artifact["file_hash"]:
            raise ValueError("rejected stance trajectory changed")
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "playmaker_pass_actor.py",
        Path(__file__).parents[1] / "growth" / "runtime_contact_target_actor.py",
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("team-compatible pass output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["default_team_compatible_pass_actions", "run_team_compatible_pass_discovery"]
