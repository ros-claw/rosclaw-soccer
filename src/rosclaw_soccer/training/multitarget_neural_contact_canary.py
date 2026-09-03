"""Causal canary for a multi-target neural contact actor on consumed contexts."""

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


def run_multitarget_neural_contact_canary(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    teacher_report_path: Path,
    actor_training_report_path: Path,
    actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Test actor-only retention against replay and no-residual baselines."""

    if not 1 <= workers <= 4:
        raise ValueError("multi-target neural contact canary workers must be in [1, 4]")
    teacher_path = teacher_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    training_path = actor_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    actor_path = actor_path.expanduser().resolve()
    actor = load_g1_neural_contact_actor(actor_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    if (
        teacher.get("schema_version") != "rosclaw_soccer.runtime_receive_contact_teacher.v1"
        or teacher.get("status") != "PASS_RUNTIME_RECEIVE_CONTACT_TEACHER"
        or training.get("status") != "PASS_NEURAL_CONTACT_DISTILLATION"
        or training.get("source_teacher_report_hash") != teacher.get("report_hash")
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or training.get("metrics", {}).get("specialist_target_velocity_xyz_mps") is not None
        or handoff.body_hash != actor.body_hash
    ):
        raise ValueError("multi-target neural contact canary lineage changed")
    selected = teacher.get("selected")
    if (
        not isinstance(selected, list)
        or len(selected) != 2
        or not all(row["quality"]["chain_passed"] for row in selected)
    ):
        raise ValueError("multi-target canary needs two successful teacher contexts")
    cases: list[
        tuple[CausalTransitionContext, PlaymakerPassProbeAction, TargetContactPlanAction, int]
    ] = []
    for row in selected:
        artifact = row["trajectory"]
        if (
            hash_bytes((teacher_path.parent / artifact["file"]).read_bytes())
            != artifact["file_hash"]
        ):
            raise ValueError("multi-target teacher trajectory changed")
        context = CausalTransitionContext(**row["context"])
        playmaker_action = PlaymakerPassProbeAction(**row["playmaker_action"])
        action_payload = dict(row["action"])
        action_payload["target_foot_velocity_xyz_mps"] = tuple(
            action_payload["target_foot_velocity_xyz_mps"]
        )
        action = TargetContactPlanAction(**action_payload)
        handoff_decision = handoff.decide(contact_policy_frame=action.contact_policy_frame)
        if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
            raise ValueError("multi-target canary handoff rejected a teacher action")
        if not actor.target_supported(action.target_foot_velocity_xyz_mps):
            raise ValueError("multi-target canary action is outside actor support")
        cases.append((context, playmaker_action, action, handoff_decision.handoff_policy_frame))
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if actor.body_hash != qualification.body_hash:
        raise ValueError("multi-target canary Body identity changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, lead_source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.multitarget_neural_contact_canary_request.v1",
        "partition": "CONSUMED_TEACHER_CONTEXT_CAUSAL_CANARY",
        "cases": [
            {
                "context": asdict(context),
                "playmaker_action": asdict(playmaker_action),
                "action": asdict(action),
                "handoff_policy_frame": handoff_frame,
            }
            for context, playmaker_action, action, handoff_frame in cases
        ],
        "teacher_report_hash": teacher["report_hash"],
        "teacher_file_hash": hash_bytes(teacher_path.read_bytes()),
        "actor_training_report_hash": training["report_hash"],
        "actor_training_file_hash": hash_bytes(training_path.read_bytes()),
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "handoff_actor_hash": handoff.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
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
            actor_path,
            output,
            case_index,
            label,
            context,
            playmaker_action,
            action,
            handoff_frame,
            quality,
            use_actor,
        )
        for case_index, (context, playmaker_action, action, handoff_frame) in enumerate(cases)
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
    rows = [_case_row(runs, index) for index in range(2)]
    gates = {
        "both_candidates_safe": all(row["candidate"]["quality"]["safe"] for row in rows),
        "both_strict_right_foot_chains": all(
            row["candidate"]["quality"]["strict_chain_passed"] for row in rows
        ),
        "both_clear_outcomes": all(
            row["candidate"]["result"]["goal_crossed"]
            or row["candidate"]["result"]["goalkeeper_save_observed"]
            for row in rows
        ),
        "exact_replay": all(row["exact_replay"] for row in rows),
        "actor_executed": all(
            row["candidate"]["actor_active"] and row["replay"]["actor_active"] for row in rows
        ),
        "actor_is_sole_contact_residual": all(
            not run["teacher_active"] and not run["scripted_contact_active"] for run in runs
        ),
        "candidate_not_worse_than_baseline": sum(
            bool(row["candidate"]["quality"]["strict_chain_passed"]) for row in rows
        )
        >= sum(bool(row["baseline"]["quality"]["strict_chain_passed"]) for row in rows),
        "all_trajectories_bound": all(
            hash_bytes((output / run["trajectory"]["file"]).read_bytes())
            == run["trajectory"]["file_hash"]
            for run in runs
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.multitarget_neural_contact_canary.v1",
        "status": (
            "PASS_MULTITARGET_NEURAL_CONTACT_CANARY"
            if all(gates.values())
            else "REJECTED_MULTITARGET_NEURAL_CONTACT_CANARY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_TEACHER_CONTEXT_CAUSAL_CANARY",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "actor_hash": actor.actor_hash,
        "metrics": {
            "case_count": len(rows),
            "candidate_strict_count": sum(
                bool(row["candidate"]["quality"]["strict_chain_passed"]) for row in rows
            ),
            "baseline_strict_count": sum(
                bool(row["baseline"]["quality"]["strict_chain_passed"]) for row in rows
            ),
            "candidate_goal_count": sum(
                bool(row["candidate"]["result"]["goal_crossed"]) for row in rows
            ),
            "candidate_save_count": sum(
                bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in rows
            ),
            "mean_candidate_shot_speed_mps": float(
                np.mean([row["candidate"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
            ),
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "canary-report.json", report)
    return report


def validate_multitarget_neural_contact_canary(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _bound_json(source)
    request_path = source.parent / "request.json"
    if (
        report.get("status") != "PASS_MULTITARGET_NEURAL_CONTACT_CANARY"
        or report.get("promotion_eligible") is not False
        or not all(report.get("gates", {}).values())
        or report.get("implementation_hash") != _implementation_hash()
        or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
    ):
        raise ValueError("multi-target neural contact canary authority is invalid")
    for row in report["rows"]:
        for key in ("candidate", "replay", "baseline"):
            artifact = row[key]["trajectory"]
            artifact_path = source.parent / artifact["file"]
            if hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("multi-target neural contact trajectory changed")
    return report


def _run(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        str,
        CausalTransitionContext,
        PlaymakerPassProbeAction,
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
        case_index,
        label,
        context,
        playmaker_action,
        action,
        handoff_frame,
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
    base = cast(dict[str, float], kwargs["passer_parameter_overrides"])
    kwargs["passer_parameter_overrides"] = {
        **base,
        "stance_offset_x": base["stance_offset_x"] + playmaker_action.stance_correction_x_m,
        "stance_offset_y": base["stance_offset_y"] + playmaker_action.stance_correction_y_m,
        "swing_speed_scale": playmaker_action.swing_speed_scale,
    }
    base_yaw = float(kwargs["passer_yaw_rad"])
    correction = playmaker_action.body_yaw_correction_rad
    kwargs["passer_yaw_rad"] = math.atan2(
        math.sin(base_yaw + correction), math.cos(base_yaw + correction)
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
            arrival_alignment_tolerance_sec=0.02,
        ),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    if use_actor:
        kwargs.update(
            shooter_neural_contact_actor_path=actor_path,
            shooter_neural_contact_policy_frame=action.contact_policy_frame,
            shooter_neural_contact_target_velocity_xyz_mps=(action.target_foot_velocity_xyz_mps),
        )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    relative = Path(f"case-{case_index:03d}") / f"{label}.npz"
    artifact_path = output / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = _save_trajectory(artifact_path, trajectory)
    artifact["file"] = str(relative)
    actor_active = np.asarray(trajectory["shooter_neural_contact_actor_active"], dtype=np.bool_)
    teacher_active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    scripted_active = np.asarray(
        trajectory["shooter_ballistic_contact_torque_active"], dtype=np.bool_
    )
    return {
        "case_index": case_index,
        "label": label,
        "actor_enabled": use_actor,
        "result": result.to_dict(),
        "quality": strict_intended_contact_quality(
            result=result,
            trajectory=trajectory,
            quality_config=quality,
            intended_contact_foot=1,
        ),
        "actor_active": bool(np.any(actor_active)),
        "teacher_active": bool(np.any(teacher_active)),
        "scripted_contact_active": bool(np.any(scripted_active)),
        "trajectory": artifact,
    }


def _case_row(runs: list[dict[str, Any]], case_index: int) -> dict[str, Any]:
    grouped = {run["label"]: run for run in runs if run["case_index"] == case_index}
    primary = grouped["candidate-primary"]
    replay = grouped["candidate-replay"]
    baseline = grouped["no-contact-residual-baseline"]
    return {
        "case_index": case_index,
        "candidate": primary,
        "replay": replay,
        "baseline": baseline,
        "exact_replay": bool(
            primary["result"] == replay["result"]
            and primary["trajectory"]["trajectory_digest"]
            == replay["trajectory"]["trajectory_digest"]
        ),
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
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
        raise ValueError("multi-target neural contact output must be new and external")
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
    "run_multitarget_neural_contact_canary",
    "validate_multitarget_neural_contact_canary",
]
