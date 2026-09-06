"""Failure-driven counterfactual search over prepared finisher plans."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import load_runtime_contact_target_actor
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    RuntimeFinishPlanAction,
    load_runtime_finish_plan_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
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


def run_prepared_finish_plan_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_path: Path,
    finish_plan_actor_path: Path,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    actions: tuple[RuntimeFinishPlanAction, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("prepared finish repair workers must be in [1, 4]")
    exam_path = rejected_exam_path.expanduser().resolve()
    exam = _bound_json(exam_path)
    exam_request = exam_path.parent / "request.json"
    actor_path = finish_plan_actor_path.expanduser().resolve()
    actor = load_runtime_finish_plan_actor(actor_path)
    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    failed_rows = [
        row
        for row in exam.get("rows", ())
        if not row["candidate"]["quality"]["strict_chain_passed"]
    ]
    learned_actions = tuple(
        sorted(
            {memory.action for memory in actor.successful_memories},
            key=lambda action: action.action_hash,
        )
    )
    active_actions = actions or learned_actions
    if (
        exam.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam.v1"
        or exam.get("status") != "REJECTED_RUNTIME_FINISH_PLAN"
        or exam.get("sealed") is not True
        or exam.get("promotion_eligible") is not False
        or not exam_request.is_file()
        or hash_bytes(exam_request.read_bytes()) != exam.get("request_hash")
        or exam.get("finish_plan_actor_hash") != actor.actor_hash
        or actor.neural_contact_actor_hash != neural.actor_hash
        or actor.contact_handoff_actor_hash != handoff.actor_hash
        or actor.body_hash != qualification.body_hash
        or base.body_hash != qualification.body_hash
        or len(failed_rows) != 4
        or not 4 <= len(active_actions) <= 12
        or len({action.action_hash for action in active_actions}) != len(active_actions)
    ):
        raise ValueError("prepared finish repair lineage changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.prepared_finish_plan_repair_request.v1",
        "partition": "CONSUMED_S155_FAILURES",
        "failed_contexts": [row["context"] for row in failed_rows],
        "failed_context_hashes": [row["context_hash"] for row in failed_rows],
        "actions": [asdict(action) for action in active_actions],
        "action_hashes": [action.action_hash for action in active_actions],
        "rejected_exam_hash": exam["report_hash"],
        "rejected_exam_file_hash": hash_bytes(exam_path.read_bytes()),
        "finish_plan_actor_hash": actor.actor_hash,
        "finish_plan_actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": actor.roster_hash,
        "finisher_self_model_hash": actor.finisher_self_model_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "implementation_hash": _implementation_hash(),
        "credit_contract": {
            "plastic_agent_id": "red.finisher",
            "plastic_skill": "prepared_receive_and_strike_plan",
            "plan_selected_before_physics_start": True,
            "neural_contact_actor_frozen": True,
        },
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            base_path,
            neural_path,
            output,
            index,
            row,
            action,
            handoff.selected_offset_frames,
            quality,
        )
        for index, (row, action) in enumerate(
            (row, action) for row in failed_rows for action in active_actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select(rows, str(source["context_hash"])) for source in failed_rows]
    selected_strict = [row for row in selected if row["quality"]["strict_chain_passed"]]
    recovered = {row["context_hash"] for row in rows if row["quality"]["strict_chain_passed"]}
    failure_route_counts: dict[str, int] = {}
    for row in rows:
        route = str(row["failure_route"])
        failure_route_counts[route] = failure_route_counts.get(route, 0) + 1
    gates = {
        "at_least_three_contexts_recovered": len(recovered) >= 3,
        "selected_safe_all": all(row["quality"]["safe"] for row in selected),
        "selected_strict_at_least_three": sum(
            bool(row["quality"]["strict_chain_passed"]) for row in selected
        )
        >= 3,
        "both_goal_and_save": any(row["result"]["goal_crossed"] for row in selected_strict)
        and any(row["result"]["goalkeeper_save_observed"] for row in selected_strict),
        "neural_actor_realized_all": all(row["neural_actor_active"] for row in rows),
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
        "schema_version": "rosclaw_soccer.prepared_finish_plan_repair.v1",
        "status": (
            "PASS_PREPARED_FINISH_PLAN_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_PREPARED_FINISH_PLAN_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_S155_FAILURES",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_hash": exam["report_hash"],
        "finish_plan_actor_hash": actor.actor_hash,
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": actor.roster_hash,
        "finisher_self_model_hash": actor.finisher_self_model_hash,
        "metrics": {
            "probe_count": len(rows),
            "context_count": len(failed_rows),
            "action_count": len(active_actions),
            "safe_count": sum(bool(row["quality"]["safe"]) for row in rows),
            "strict_success_count": sum(
                bool(row["quality"]["strict_chain_passed"]) for row in rows
            ),
            "strict_goal_count": sum(
                bool(row["quality"]["strict_chain_passed"] and row["result"]["goal_crossed"])
                for row in rows
            ),
            "strict_save_count": sum(
                bool(
                    row["quality"]["strict_chain_passed"]
                    and row["result"]["goalkeeper_save_observed"]
                )
                for row in rows
            ),
            "recovered_context_count": len(recovered),
            "failure_route_counts": failure_route_counts,
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
    _write_json(output / "repair-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        Path,
        int,
        dict[str, Any],
        RuntimeFinishPlanAction,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        base_path,
        neural_path,
        output,
        index,
        source,
        action,
        handoff_offset,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    base = load_runtime_contact_target_actor(base_path)
    context = _context_from_dict(source["context"])
    playmaker = PlaymakerPassProbeAction(**source["playmaker_action"])
    kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=context,
        playmaker_action=playmaker,
        contact_actor_path=neural_path,
        target_actor=base,
        target_actor_path=None,
        target_velocity=action.target.target_foot_velocity_xyz_mps,
        handoff_frame=action.receive.contact_policy_frame + handoff_offset,
    )
    kwargs.update(
        shooter_parameter_overrides={
            "stance_offset_x": action.receive.stance_offset_x_m,
            "stance_offset_y": action.receive.stance_offset_y_m,
            "foot_yaw_offset": action.receive.foot_yaw_offset_rad,
            "foot_pitch_offset": action.receive.foot_pitch_offset_rad,
        },
        shooter_causal_strike_option_config=replace(
            kwargs["shooter_causal_strike_option_config"],
            maximum_arrival_advance_frames=action.receive.maximum_arrival_advance_frames,
            arrival_alignment_tolerance_sec=action.receive.arrival_alignment_tolerance_sec,
        ),
        shooter_neural_contact_policy_frame=action.receive.contact_policy_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    relative = Path(f"probe-{index:03d}.npz")
    artifact = _save_trajectory(output / relative, trajectory)
    artifact["file"] = str(relative)
    measured_result = result.to_dict()
    measured_quality = strict_intended_contact_quality(
        result=result,
        trajectory=trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    return {
        "probe_index": index,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "playmaker_action": asdict(playmaker),
        "action": asdict(action),
        "action_hash": action.action_hash,
        "result": measured_result,
        "quality": measured_quality,
        "failure_route": _failure_route(measured_quality, measured_result),
        "neural_actor_active": bool(np.any(trajectory["shooter_neural_contact_actor_active"])),
        "teacher_active": bool(np.any(trajectory["shooter_loft_teacher_active"])),
        "scripted_contact_active": bool(
            np.any(trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "trajectory": artifact,
    }


def _select(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    return min(
        (row for row in rows if row["context_hash"] == context_hash),
        key=lambda row: (
            not row["quality"]["strict_chain_passed"],
            not row["quality"]["safe"],
            not row["quality"]["intended_foot_contact"],
            not row["quality"]["clear_outcome"],
            -float(row["result"]["shot_peak_ball_speed_mps"]),
            row["action_hash"],
        ),
    )


def _failure_route(quality: dict[str, Any], result: dict[str, Any]) -> str:
    if quality["strict_chain_passed"]:
        return "SUCCESS"
    if not quality["safe"]:
        return "POST_CONTACT_STABILITY"
    if not result["pass_contact_observed"] or not result["shot_contact_observed"]:
        return "PHASE_OR_CONTACT_ACQUISITION"
    if not quality["intended_foot_contact"]:
        return "INTENDED_FOOT_ALIGNMENT"
    if not quality["clear_outcome"]:
        return "SHOT_DIRECTION"
    if not quality["chain_passed"]:
        return "SHOT_SPEED_OR_ORDERING"
    return "UNCLASSIFIED"


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
        raise ValueError("prepared finish repair output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["run_prepared_finish_plan_repair"]
