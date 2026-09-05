"""Consume a rejected RECEIVE holdout as the next failure-driven curriculum."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_receive_actor import load_runtime_receive_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.runtime_receive_discovery import (
    _run_probe,
    _select,
    default_runtime_receive_actions,
)


def run_runtime_receive_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_report_path: Path,
    parent_receive_actor_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("runtime RECEIVE repair workers must be in [1, 4]")
    exam_path = rejected_exam_report_path.expanduser().resolve()
    exam = _bound_report(exam_path)
    exam_request = _read_object(exam_path.parent / "request.json")
    parent_path = parent_receive_actor_path.expanduser().resolve()
    parent = load_runtime_receive_actor(parent_path)
    finisher_path = finisher_actor_path.expanduser().resolve()
    finisher = load_g1_neural_contact_actor(finisher_path)
    teacher = _bound_json(teacher_discovery_report_path.expanduser().resolve())
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    case_payloads = exam_request.get("cases")
    if (
        exam.get("schema_version") != "rosclaw_soccer.runtime_receive_exam.v1"
        or exam.get("status") != "REJECTED_RUNTIME_RECEIVE"
        or exam.get("promotion_eligible") is not False
        or not isinstance(case_payloads, list)
        or len(case_payloads) != 6
        or exam.get("actor_hash") != parent.actor_hash
        or exam.get("roster_hash") != parent.roster_hash
        or exam.get("finisher_self_model_hash") != parent.finisher_self_model_hash
        or finisher.body_hash != qualification.body_hash
        or parent.body_hash != qualification.body_hash
        or parent.kick_prior_hash != qualification.kick_prior_hash
        or handoff.body_hash != qualification.body_hash
    ):
        raise ValueError("runtime RECEIVE repair lineage changed")
    for index, row in enumerate(exam["rows"]):
        case_dir = exam_path.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = row[key]
            if hash_bytes((case_dir / artifact["file"]).read_bytes()) != artifact["file_hash"]:
                raise ValueError("runtime RECEIVE rejected trajectory changed")
    contexts = tuple(CausalTransitionContext(**value["context"]) for value in case_payloads)
    playmaker_actions = tuple(
        PlaymakerPassProbeAction(**value["playmaker_action"]) for value in case_payloads
    )
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("runtime RECEIVE repair needs one frozen contact target")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    contact_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=contact_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime RECEIVE repair handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    actions = default_runtime_receive_actions()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_repair_request.v2",
        "partition": "CONSUMED_REJECTED_S134_HOLDOUT",
        "plastic_agent_id": "red.finisher",
        "plastic_role": "finisher",
        "plastic_intent": "receive",
        "plastic_skill": "first_touch",
        "cases": case_payloads,
        "context_hashes": [context.context_hash for context in contexts],
        "actions": [asdict(action) for action in actions],
        "action_hashes": [action.action_hash for action in actions],
        "rejected_exam_report_hash": exam["report_hash"],
        "rejected_exam_file_hash": hash_bytes(exam_path.read_bytes()),
        "parent_actor_hash": parent.actor_hash,
        "parent_actor_file_hash": hash_bytes(parent_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "frozen_finisher_actor_hash": finisher.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(finisher_path.read_bytes()),
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "frozen_handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "roster_hash": parent.roster_hash,
        "finisher_self_model_hash": parent.finisher_self_model_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "contact_action_hash": contact_action.action_hash,
        "intervention_contract": {
            "observation_precedes_action": True,
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "initial_receive_ready_action_hash": contact_action.action_hash,
            "enumerated_action_applied_before_decision": False,
        },
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
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
        "at_least_four_failed_contexts_recovered": len(recovered) >= 4,
        "all_selected_safe": all(row["quality"]["safe"] for row in selected),
        "selected_recovered_when_available": all(
            row["strict_team_chain"] for row in selected if row["context_hash"] in recovered
        ),
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
        "schema_version": "rosclaw_soccer.runtime_receive_repair.v2",
        "status": (
            "PASS_RUNTIME_RECEIVE_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_RUNTIME_RECEIVE_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "claim": "FAILURE_DRIVEN_ROLE_LOCAL_RECEIVE_SUPPORT_EXPANSION",
        "intervention_contract": request["intervention_contract"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "plastic_agent_id": "red.finisher",
        "roster_hash": parent.roster_hash,
        "finisher_self_model_hash": parent.finisher_self_model_hash,
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
    _write_json(output / "repair-report.json", report)
    return report


def _bound_report(path: Path) -> dict[str, Any]:
    report = _read_object(path)
    claimed = report.pop("report_hash", None)
    if claimed != hash_json(report):
        raise ValueError("runtime RECEIVE rejected report integrity changed")
    report["report_hash"] = claimed
    request = path.parent / "request.json"
    if not request.is_file() or hash_bytes(request.read_bytes()) != report.get("request_hash"):
        raise ValueError("runtime RECEIVE rejected request binding changed")
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_receive_actor.py",
        Path(__file__).parents[1] / "growth" / "causal_strike_option.py",
        Path(__file__).parents[1] / "growth" / "role_self_model.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
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
        raise ValueError("runtime RECEIVE repair output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["run_runtime_receive_repair"]
