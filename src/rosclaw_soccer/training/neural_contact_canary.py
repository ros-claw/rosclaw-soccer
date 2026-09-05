"""Sealed causal canary for the complete neural contact muscle memory."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionGrowthConfig,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)


def run_neural_contact_canary(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    teacher_discovery_report_path: Path,
    actor_training_report_path: Path,
    actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 3,
) -> dict[str, Any]:
    if not 1 <= workers <= 3:
        raise ValueError("neural contact canary workers must be in [1, 3]")
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    training_path = actor_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    actor = load_g1_neural_contact_actor(actor_path)
    handoff = load_contact_handoff_actor(handoff_actor_path)
    if (
        teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        or training.get("status") != "PASS_NEURAL_CONTACT_DISTILLATION"
        or training.get("source_teacher_report_hash") != teacher["report_hash"]
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or not training.get("replaces_scripted_contact_torque")
        or handoff.body_hash != actor.body_hash
    ):
        raise ValueError("neural contact canary lineage changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("neural contact canary expects one isolated teacher success")
    source_row = source_rows[0]
    source_artifact = source_row["trajectory"]
    source_trajectory = teacher_path.parent / source_artifact["file"]
    if hash_bytes(source_trajectory.read_bytes()) != source_artifact["file_hash"]:
        raise ValueError("neural contact canary source trajectory changed")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if actor.body_hash != qualification.body_hash:
        raise ValueError("neural contact canary Body identity changed")
    context = _context_from_dict(source_row["context"])
    action_payload = dict(source_row["action"])
    action_payload["target_foot_velocity_xyz_mps"] = tuple(
        action_payload["target_foot_velocity_xyz_mps"]
    )
    action = TargetContactPlanAction(**action_payload)
    handoff_decision = handoff.decide(contact_policy_frame=action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("neural contact canary handoff rejected the source action")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_canary_request.v1",
        "partition": "CONSUMED_TEACHER_SUCCESS_CAUSAL_CANARY",
        "context": asdict(context),
        "context_hash": context.context_hash,
        "action": asdict(action),
        "teacher_discovery_report_hash": teacher["report_hash"],
        "teacher_discovery_file_hash": hash_bytes(teacher_path.read_bytes()),
        "teacher_source_trajectory_hash": source_artifact["file_hash"],
        "actor_training_report_hash": training["report_hash"],
        "actor_training_file_hash": hash_bytes(training_path.read_bytes()),
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "contact_handoff_actor_hash": handoff.actor_hash,
        "contact_handoff_actor_file_hash": hash_bytes(handoff_actor_path.read_bytes()),
        "handoff_decision": asdict(handoff_decision),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "intended_contact_foot": "right",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            actor_path.expanduser().resolve(),
            output,
            label,
            context,
            action,
            handoff_decision.handoff_policy_frame,
            quality,
            use_actor,
        )
        for label, use_actor in (
            ("candidate-primary", True),
            ("candidate-replay", True),
            ("no-contact-residual-baseline", False),
        )
    )
    if workers == 1:
        runs = [_run(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = list(executor.map(_run, jobs))
    primary, replay, baseline = runs
    candidate_result = primary["result"]
    candidate_quality = primary["quality"]
    baseline_result = baseline["result"]
    exact_replay = bool(
        primary["result"] == replay["result"]
        and primary["trajectory"]["trajectory_digest"] == replay["trajectory"]["trajectory_digest"]
    )
    gates = {
        "candidate_safe": bool(candidate_quality["safe"]),
        "strict_right_foot_chain": bool(candidate_quality["strict_chain_passed"]),
        "candidate_goal": bool(candidate_result["goal_crossed"]),
        "minimum_shot_speed": candidate_result["shot_peak_ball_speed_mps"] >= 9.5,
        "exact_replay": exact_replay,
        "actor_executed": primary["actor_active"] and replay["actor_active"],
        "actor_sole_contact_residual": all(
            not run["teacher_active"] and not run["scripted_contact_active"] for run in runs
        ),
        "challenging_baseline_saved": bool(
            baseline["quality"]["strict_chain_passed"]
            and baseline_result["goalkeeper_save_observed"]
            and not baseline_result["goal_crossed"]
        ),
        "causal_goal_delta": bool(
            candidate_result["goal_crossed"] and not baseline_result["goal_crossed"]
        ),
        "stability_retained": bool(
            candidate_result["shooter_min_pelvis_height_m"]
            >= baseline_result["shooter_min_pelvis_height_m"] - 0.02
            and candidate_result["shooter_tail_wobble_index"]
            <= baseline_result["shooter_tail_wobble_index"] + 0.02
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_canary.v1",
        "status": (
            "PASS_NEURAL_CONTACT_CANARY"
            if all(gates.values())
            else "REJECTED_NEURAL_CONTACT_CANARY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_TEACHER_SUCCESS_CAUSAL_CANARY",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "actor_hash": actor.actor_hash,
        "contact_handoff_actor_hash": handoff.actor_hash,
        "metrics": {
            "candidate_shot_speed_mps": candidate_result["shot_peak_ball_speed_mps"],
            "baseline_shot_speed_mps": baseline_result["shot_peak_ball_speed_mps"],
            "candidate_minimum_pelvis_height_m": candidate_result["shooter_min_pelvis_height_m"],
            "candidate_tail_wobble_index": candidate_result["shooter_tail_wobble_index"],
            "candidate_support_slip_m": candidate_result[
                "shooter_post_contact_support_foot_slip_m"
            ],
        },
        "gates": gates,
        "runs": runs,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "canary-report.json", report)
    return report


def _run(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        str,
        Any,
        TargetContactPlanAction,
        int,
        CausalTransitionGrowthConfig,
        bool,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_dir,
        actor_path,
        output,
        label,
        context,
        action,
        handoff,
        quality,
        use_actor,
    ) = job
    lead, _ = _load_lead_policy(source_dir)
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    kwargs.update(
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
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff,
    )
    if use_actor:
        kwargs.update(
            shooter_neural_contact_actor_path=actor_path,
            shooter_neural_contact_policy_frame=action.contact_policy_frame,
            shooter_neural_contact_target_velocity_xyz_mps=(action.target_foot_velocity_xyz_mps),
        )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"{label}.npz", trajectory)
    active = np.asarray(trajectory["shooter_neural_contact_actor_active"], dtype=np.bool_)
    teacher = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    scripted = np.asarray(trajectory["shooter_ballistic_contact_torque_active"], dtype=np.bool_)
    return {
        "label": label,
        "actor_enabled": use_actor,
        "result": result.to_dict(),
        "quality": strict_intended_contact_quality(
            result=result,
            trajectory=trajectory,
            quality_config=quality,
            intended_contact_foot=1,
        ),
        "actor_active": bool(np.any(active)),
        "actor_active_frame_count": int(np.count_nonzero(active)),
        "teacher_active": bool(np.any(teacher)),
        "scripted_contact_active": bool(np.any(scripted)),
        "trajectory": artifact,
    }


def validate_neural_contact_canary(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _bound_json(source)
    if (
        report.get("status") != "PASS_NEURAL_CONTACT_CANARY"
        or report.get("promotion_eligible") is not False
        or not all(report.get("gates", {}).values())
        or report.get("implementation_hash") != _implementation_hash()
    ):
        raise ValueError("neural contact canary authority is invalid")
    if hash_bytes((source.parent / "request.json").read_bytes()) != report["request_hash"]:
        raise ValueError("neural contact canary request changed")
    for run in report["runs"]:
        artifact = run["trajectory"]
        trajectory_path = source.parent / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("neural contact canary trajectory changed")
    return report


def _bound_json(path: Path) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    claimed = payload.pop("report_hash", None)
    if claimed != hash_json(payload):
        raise ValueError("bound report integrity changed")
    payload["report_hash"] = claimed
    return payload


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "growth" / "contact_handoff_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("neural contact canary output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["run_neural_contact_canary", "validate_neural_contact_canary"]
