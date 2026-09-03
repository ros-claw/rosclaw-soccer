"""Failure-driven discovery for the finisher's runtime RECEIVE control law."""

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
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
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
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction


def default_runtime_receive_actions() -> tuple[RuntimeReceiveAction, ...]:
    """Preregister the local control-law probes before new evidence is run."""

    return (
        RuntimeReceiveAction(
            arrival_alignment_tolerance_sec=0.08,
            stance_offset_y_m=-0.06,
            foot_yaw_offset_rad=-0.04,
        ),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, foot_yaw_offset_rad=-0.04),
        RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=-0.12),
        RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=-0.04),
        RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=0.0),
        RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=0.04),
        RuntimeReceiveAction(stance_offset_y_m=-0.04, foot_yaw_offset_rad=0.08),
        RuntimeReceiveAction(stance_offset_y_m=-0.02, foot_yaw_offset_rad=-0.04),
    )


def runtime_receive_timing_actions() -> tuple[RuntimeReceiveAction, ...]:
    """Probe deployable late-phase timing while receive-ready posture stays fixed."""

    return (
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=244),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=248),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=250),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=252),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=254),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=256),
        RuntimeReceiveAction(stance_offset_y_m=-0.06, contact_policy_frame=258),
        RuntimeReceiveAction(
            stance_offset_y_m=-0.06,
            contact_policy_frame=252,
            foot_yaw_offset_rad=-0.06,
        ),
    )


def runtime_receive_direction_actions() -> tuple[RuntimeReceiveAction, ...]:
    """Probe bounded foot direction at the best late intervention frame."""

    return tuple(
        RuntimeReceiveAction(
            stance_offset_y_m=-0.06,
            contact_policy_frame=258,
            foot_yaw_offset_rad=yaw,
        )
        for yaw in (-0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    )


def run_runtime_receive_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_handshake_report_path: Path,
    role_audit_report_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    probe_actions: tuple[RuntimeReceiveAction, ...] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Search only red.finisher RECEIVE while its team-mates stay frozen."""

    if not 1 <= workers <= 4:
        raise ValueError("runtime RECEIVE discovery workers must be in [1, 4]")
    handshake_path = rejected_handshake_report_path.expanduser().resolve()
    handshake = _bound_historical_report(handshake_path)
    handshake_request = _read_object(handshake_path.parent / "request.json")
    role_path = role_audit_report_path.expanduser().resolve()
    role_audit = _bound_historical_report(role_path)
    role_request = role_path.parent / "request.json"
    finisher_path = finisher_actor_path.expanduser().resolve()
    finisher = load_g1_neural_contact_actor(finisher_path)
    teacher = _bound_json(teacher_discovery_report_path.expanduser().resolve())
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    selected_source = handshake.get("selected")
    if (
        handshake.get("schema_version") != "rosclaw_soccer.team_pass_handshake_discovery.v1"
        or handshake.get("status") != "REJECTED_TEAM_PASS_HANDSHAKE_DISCOVERY"
        or not isinstance(selected_source, list)
        or len(selected_source) != 2
        or handshake_request.get("frozen_finisher_actor_hash") != finisher.actor_hash
        or handshake_request.get("frozen_handoff_actor_hash") != handoff.actor_hash
        or finisher.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or role_audit.get("status") != "PASS_ROLE_AWARENESS_AND_CREDIT_AUDIT"
        or hash_json(role_audit.get("roster")) != role_audit.get("roster_hash")
        or not role_request.is_file()
        or hash_bytes(role_request.read_bytes()) != role_audit.get("request_hash")
        or teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
    ):
        raise ValueError("runtime RECEIVE discovery lineage changed")
    for row in handshake["rows"]:
        artifact = row["trajectory"]
        if (
            hash_bytes((handshake_path.parent / artifact["file"]).read_bytes())
            != artifact["file_hash"]
        ):
            raise ValueError("runtime RECEIVE source trajectory changed")
    finisher_models = [
        item
        for item in cast(dict[str, Any], role_audit["roster"])["agents"]
        if item["agent_id"] == "red.finisher"
    ]
    if len(finisher_models) != 1:
        raise ValueError("runtime RECEIVE role audit lacks one red.finisher")
    finisher_self_model_hash = hash_json(finisher_models[0])

    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("runtime RECEIVE needs one frozen neural contact target")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    contact_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=contact_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime RECEIVE handoff is invalid")

    contexts = tuple(_context_from_dict(row["context"]) for row in selected_source)
    playmaker_actions = tuple(
        PlaymakerPassProbeAction(**row["probe"]["playmaker_action"]) for row in selected_source
    )
    actions = probe_actions or default_runtime_receive_actions()
    if len(actions) != 8 or len({action.action_hash for action in actions}) != 8:
        raise ValueError("runtime RECEIVE discovery requires eight unique actions")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_discovery_request.v2",
        "partition": "CONSUMED_S133_FAILED_TEAM_HANDSHAKES",
        "plastic_agent_id": "red.finisher",
        "plastic_role": "finisher",
        "plastic_intent": "receive",
        "plastic_skill": "first_touch",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "frozen_playmaker_actions": [asdict(action) for action in playmaker_actions],
        "actions": [asdict(action) for action in actions],
        "action_hashes": [action.action_hash for action in actions],
        "source_handshake_report_hash": handshake["report_hash"],
        "source_handshake_file_hash": hash_bytes(handshake_path.read_bytes()),
        "source_handshake_implementation_hash": handshake["implementation_hash"],
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "role_audit_report_hash": role_audit["report_hash"],
        "roster_hash": role_audit["roster_hash"],
        "finisher_self_model_hash": finisher_self_model_hash,
        "frozen_finisher_actor_hash": finisher.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(finisher_path.read_bytes()),
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "frozen_handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "frozen_goalkeeper_policy_hash": handshake_request["frozen_goalkeeper_policy_hash"],
        "contact_action_hash": contact_action.action_hash,
        "intervention_contract": {
            "observation_precedes_action": True,
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "initial_receive_ready_action_hash": contact_action.action_hash,
            "enumerated_action_applied_before_decision": False,
        },
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
            finisher_path,
            output,
            index,
            context,
            playmaker_action,
            action,
            contact_action,
            handoff_decision.handoff_policy_frame,
            quality,
        )
        for index, (context, playmaker_action, action) in enumerate(
            (context, playmaker_action, action)
            for context, playmaker_action in zip(contexts, playmaker_actions, strict=True)
            for action in actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select(rows, context.context_hash) for context in contexts]
    recovered = {row["context_hash"] for row in rows if row["strict_team_chain"]}
    gates = {
        "all_failed_contexts_recovered": recovered == {item.context_hash for item in contexts},
        "all_selected_safe": all(row["quality"]["safe"] for row in selected),
        "all_selected_strict_team_chain": all(row["strict_team_chain"] for row in selected),
        "all_selected_clear_outcome": all(row["quality"]["clear_outcome"] for row in selected),
        "neural_contact_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "no_teacher_or_scripted_contact": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "runtime_intervention_observed_all": all(
            row["runtime_intervention_observed"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_discovery.v2",
        "status": (
            "PASS_RUNTIME_RECEIVE_DISCOVERY"
            if all(gates.values())
            else "REJECTED_RUNTIME_RECEIVE_DISCOVERY"
        ),
        "promotion_eligible": False,
        "claim": "ROLE_LOCAL_RUNTIME_RECEIVE_REPAIR_WITH_TEAMMATES_AND_OPPONENT_FROZEN",
        "intervention_contract": request["intervention_contract"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "plastic_agent_id": "red.finisher",
        "roster_hash": role_audit["roster_hash"],
        "finisher_self_model_hash": finisher_self_model_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "metrics": {
            "context_count": len(contexts),
            "probe_count": len(rows),
            "safe_probe_count": sum(int(row["quality"]["safe"]) for row in rows),
            "strict_team_chain_count": sum(int(row["strict_team_chain"]) for row in rows),
            "recovered_context_count": len(recovered),
            "goal_count": sum(int(row["result"]["goal_crossed"]) for row in rows),
            "save_count": sum(int(row["result"]["goalkeeper_save_observed"]) for row in rows),
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
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
        int,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
        RuntimeReceiveAction,
        TargetContactPlanAction,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        actor_path,
        output,
        index,
        context,
        playmaker_action,
        action,
        contact_action,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_s95_dir)
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
            maximum_arrival_advance_frames=contact_action.maximum_arrival_advance_frames,
        ),
        shooter_runtime_receive_probe_action=action,
        shooter_neural_contact_actor_path=actor_path,
        shooter_neural_contact_policy_frame=contact_action.contact_policy_frame,
        shooter_neural_contact_target_velocity_xyz_mps=(
            contact_action.target_foot_velocity_xyz_mps
        ),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    quality_result = strict_intended_contact_quality(
        result=result,
        trajectory=trajectory,
        quality_config=quality,
        intended_contact_foot=1,
    )
    return {
        "probe_index": index,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "playmaker_action": asdict(playmaker_action),
        "action": asdict(action),
        "action_hash": action.action_hash,
        "result": result.to_dict(),
        "quality": quality_result,
        "strict_team_chain": bool(quality_result["strict_chain_passed"]),
        "runtime_intervention_observed": bool(
            result.shooter_runtime_receive_route == "TRAINING_RUNTIME_RECEIVE_INTERVENTION"
        ),
        "finisher_actor_active": bool(np.any(trajectory["shooter_neural_contact_actor_active"])),
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
            not row["strict_team_chain"],
            not row["quality"]["safe"],
            not row["quality"]["clear_outcome"],
            -float(row["result"]["shot_peak_ball_speed_mps"]),
            row["action_hash"],
        ),
    )


def _bound_historical_report(path: Path) -> dict[str, Any]:
    report = _read_object(path)
    claimed = report.pop("report_hash", None)
    if claimed != hash_json(report):
        raise ValueError("historical report integrity changed")
    report["report_hash"] = claimed
    request = path.parent / "request.json"
    if not request.is_file() or hash_bytes(request.read_bytes()) != report.get("request_hash"):
        raise ValueError("historical report request binding changed")
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_receive_actor.py",
        Path(__file__).parents[1] / "growth" / "causal_strike_option.py",
        Path(__file__).parents[1] / "growth" / "role_self_model.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
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
        raise ValueError("runtime RECEIVE output must use a new external directory")
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
    "default_runtime_receive_actions",
    "run_runtime_receive_discovery",
    "runtime_receive_direction_actions",
    "runtime_receive_timing_actions",
]
