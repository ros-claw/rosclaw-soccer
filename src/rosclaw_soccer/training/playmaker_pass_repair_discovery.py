"""Feed a rejected fresh playmaker exam back into role-local practice."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.playmaker_pass_actor import load_playmaker_pass_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionGrowthConfig
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import (
    PlaymakerPassProbeAction,
    _run_probe,
)
from rosclaw_soccer.training.playmaker_pass_holdout_exam import (
    validate_playmaker_pass_holdout_exam,
)


def playmaker_repair_actions() -> tuple[PlaymakerPassProbeAction, ...]:
    """Preregister a compact response-informed search without stance rewrites."""

    actions = (
        PlaymakerPassProbeAction(),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.06),
        PlaymakerPassProbeAction(body_yaw_correction_rad=-0.02),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.04, swing_speed_scale=0.84),
        PlaymakerPassProbeAction(body_yaw_correction_rad=0.06, swing_speed_scale=0.84),
    )
    if len({action.action_hash for action in actions}) != len(actions):
        raise RuntimeError("playmaker repair search contains duplicate actions")
    return actions


def run_playmaker_pass_repair_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_holdout_report_path: Path,
    playmaker_actor_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Search corrections on every failed S132 fresh context."""

    if not 1 <= workers <= 4:
        raise ValueError("playmaker repair workers must be in [1, 4]")
    rejected_path = rejected_holdout_report_path.expanduser().resolve()
    rejected = validate_playmaker_pass_holdout_exam(rejected_path)
    playmaker_source = playmaker_actor_path.expanduser().resolve()
    finisher_source = finisher_actor_path.expanduser().resolve()
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    handoff_source = handoff_actor_path.expanduser().resolve()
    playmaker = load_playmaker_pass_actor(playmaker_source)
    finisher = load_g1_neural_contact_actor(finisher_source)
    teacher = _bound_json(teacher_path)
    handoff = load_contact_handoff_actor(handoff_source)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        rejected.get("status") != "REJECTED_PLAYMAKER_PASS_FRESH_HOLDOUT"
        or rejected.get("actor_hash") != playmaker.actor_hash
        or playmaker.frozen_finisher_actor_hash != finisher.actor_hash
        or playmaker.body_hash != qualification.body_hash
        or finisher.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
    ):
        raise ValueError("playmaker repair lineage changed")
    request_source = _read_object(rejected_path.parent / "request.json")
    if hash_bytes((rejected_path.parent / "request.json").read_bytes()) != rejected["request_hash"]:
        raise ValueError("playmaker rejected holdout request changed")
    contexts = tuple(_context_from_dict(value) for value in request_source["contexts"])
    failed_hashes = {row["context_hash"] for row in rejected["rows"] if not row["delivery_passed"]}
    contexts = tuple(context for context in contexts if context.context_hash in failed_hashes)
    if len(contexts) != 3:
        raise ValueError("playmaker repair expects all three failed fresh contexts")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("playmaker repair needs one frozen finisher action")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    finisher_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=finisher_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("playmaker repair finisher handoff is invalid")
    actions = playmaker_repair_actions()
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_repair_request.v1",
        "partition": "CONSUMED_S132_ROLE_LOCAL_FAILURES",
        "plastic_agent_id": "red.playmaker",
        "plastic_skill": "lead_pass",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "actions": [asdict(action) for action in actions],
        "action_hashes": [action.action_hash for action in actions],
        "rejected_holdout_report_hash": rejected["report_hash"],
        "playmaker_actor_hash": playmaker.actor_hash,
        "playmaker_actor_file_hash": hash_bytes(playmaker_source.read_bytes()),
        "frozen_finisher_actor_hash": finisher.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(finisher_source.read_bytes()),
        "frozen_goalkeeper_policy_hash": playmaker.frozen_goalkeeper_policy_hash,
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "maximum_delivery_error_m": playmaker.maximum_delivery_error_m,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
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
            finisher_source,
            output,
            index,
            context,
            action,
            finisher_action,
            handoff_decision.handoff_policy_frame,
            quality,
            playmaker.maximum_delivery_error_m,
        )
        for index, (context, action) in enumerate(
            (context, action) for context in contexts for action in actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select(rows, context.context_hash) for context in contexts]
    recovered = {row["context_hash"] for row in rows if row["playmaker_pass_success"]}
    gates = {
        "all_failed_contexts_recovered": recovered
        == {context.context_hash for context in contexts},
        "all_selected_safe": all(row["quality"]["safe"] for row in selected),
        "all_selected_ordered": all(row["quality"]["ordered_contacts"] for row in selected),
        "all_selected_delivery_passed": all(row["playmaker_pass_success"] for row in selected),
        "minimum_two_strict_complete_chains": sum(
            int(row["quality"]["strict_chain_passed"]) for row in selected
        )
        >= 2,
        "finisher_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "no_teacher_or_scripted_contact": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_repair_discovery.v1",
        "status": (
            "PASS_PLAYMAKER_PASS_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_PLAYMAKER_PASS_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "claim": "FAILURE_FED_ROLE_LOCAL_REPAIR_WITH_TEAMMATE_AND_OPPONENT_FROZEN",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "metrics": {
            "context_count": len(contexts),
            "probe_count": len(rows),
            "safe_probe_count": sum(int(row["quality"]["safe"]) for row in rows),
            "recovered_context_count": len(recovered),
            "strict_selected_count": sum(
                int(row["quality"]["strict_chain_passed"]) for row in selected
            ),
            "selected_mean_delivery_error_m": float(np.mean([_error(row) for row in selected])),
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "repair-report.json", report)
    return report


def _select(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["context_hash"] == context_hash]
    return min(
        candidates,
        key=lambda row: (
            not row["quality"]["safe"],
            not row["quality"]["ordered_contacts"],
            _error(row),
            not row["quality"]["strict_chain_passed"],
            row["action_hash"],
        ),
    )


def _error(row: dict[str, Any]) -> float:
    value = row["result"].get("pass_delivery_error_m")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 5.0


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parent / "playmaker_pass_discovery.py",
        Path(__file__).parents[1] / "growth" / "playmaker_pass_actor.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("playmaker repair output must use a new external directory")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["playmaker_repair_actions", "run_playmaker_pass_repair_discovery"]
