"""Discover stable stance alignment for the actor-controlled strike foot."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.growth.target_velocity_contact_actor import (
    load_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _chain_quality,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict


@dataclass(frozen=True)
class IntendedFootAlignmentProbe:
    context: CausalTransitionContext
    action: TargetContactPlanAction
    intended_contact_foot: int = 1

    def __post_init__(self) -> None:
        if self.intended_contact_foot not in {-1, 1}:
            raise ValueError("intended contact foot must be left or right")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_intended_foot_alignment_probes(
    contexts: tuple[CausalTransitionContext, ...],
) -> tuple[IntendedFootAlignmentProbe, ...]:
    if len(contexts) != 6:
        raise ValueError("intended-foot alignment needs six consumed contexts")
    lateral_stances = (-0.12, 0.0, 0.12)
    probes: list[IntendedFootAlignmentProbe] = []
    for index, context in enumerate(contexts):
        for stance_y in lateral_stances:
            action = (
                TargetContactPlanAction(0, 0.04, stance_y, 248, 0.04, 0.01, (7.0, 1.0, 3.0))
                if index < 3
                else TargetContactPlanAction(12, -0.12, stance_y, 248, -0.04, 0.01, (7.0, 1.0, 3.0))
            )
            probes.append(IntendedFootAlignmentProbe(context, action))
    return tuple(probes)


def strict_intended_contact_quality(
    *,
    result: Any,
    trajectory: dict[str, np.ndarray],
    quality_config: CausalTransitionGrowthConfig,
    intended_contact_foot: int,
) -> dict[str, Any]:
    quality = _chain_quality(result, trajectory, quality_config)
    contact = np.asarray(trajectory["shooter_ball_contact_foot"], dtype=np.int64)
    observed = contact[contact != 0]
    first_foot = None if len(observed) == 0 else int(observed[0])
    intended = first_foot == intended_contact_foot
    return {
        **quality,
        "first_shooter_contact_foot": first_foot,
        "intended_contact_foot": intended_contact_foot,
        "intended_foot_contact": intended,
        "strict_chain_passed": bool(quality["chain_passed"] and intended),
    }


def run_intended_foot_alignment_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_adaptive_teacher_path: Path,
    target_contact_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    probes: tuple[IntendedFootAlignmentProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("intended-foot alignment workers must be in [1, 8]")
    rejected_path = rejected_adaptive_teacher_path.expanduser().resolve()
    rejected = _load_rejected_teacher(rejected_path)
    unique_contexts: dict[str, CausalTransitionContext] = {}
    for row in rejected["rows"]:
        context = _context_from_dict(row["context"])
        unique_contexts.setdefault(context.context_hash, context)
    contexts = tuple(unique_contexts.values())
    active_probes = probes or default_intended_foot_alignment_probes(contexts)
    if (
        len(active_probes) < 18
        or len({probe.probe_hash for probe in active_probes}) != len(active_probes)
        or {probe.context.context_hash for probe in active_probes} != set(unique_contexts)
    ):
        raise ValueError("intended-foot probes lack consumed context diversity")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    contact = load_g1_target_velocity_contact_actor(target_contact_actor_path)
    handoff = load_contact_handoff_actor(handoff_actor_path)
    if (
        {contact.body_hash, handoff.body_hash} != {qualification.body_hash}
        or handoff.target_contact_actor_hash != contact.actor_hash
        or rejected["contact_handoff_actor_hash"] != handoff.actor_hash
    ):
        raise ValueError("intended-foot alignment actor lineage changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.intended_foot_alignment_request.v1",
        "partition": "CONSUMED_REJECTED_HOLDOUT_STRIKE_FOOT_DEVELOPMENT",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "rejected_adaptive_teacher_report_hash": rejected["report_hash"],
        "rejected_adaptive_teacher_file_hash": hash_bytes(rejected_path.read_bytes()),
        "target_contact_actor_hash": contact.actor_hash,
        "target_contact_actor_file_hash": hash_bytes(target_contact_actor_path.read_bytes()),
        "contact_handoff_actor_hash": handoff.actor_hash,
        "contact_handoff_actor_file_hash": hash_bytes(handoff_actor_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
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
            target_contact_actor_path.expanduser().resolve(),
            handoff_actor_path.expanduser().resolve(),
            output,
            index,
            probe,
            quality,
        )
        for index, probe in enumerate(active_probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    strict_success = sum(bool(row["quality"]["strict_chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    intended_count = sum(bool(row["quality"]["intended_foot_contact"]) for row in rows)
    intended_contexts = {
        row["context_hash"] for row in rows if row["quality"]["intended_foot_contact"]
    }
    goal_count = sum(
        bool(row["quality"]["strict_chain_passed"] and row["result"]["goal_crossed"])
        for row in rows
    )
    save_count = sum(
        bool(row["quality"]["strict_chain_passed"] and row["result"]["goalkeeper_save_observed"])
        for row in rows
    )
    gates = {
        "all_safe": safe_count == len(rows),
        "intended_contact_majority": intended_count >= 12,
        "intended_context_coverage": len(intended_contexts) >= 5,
        "minimum_strict_success": strict_success >= 3,
        "both_goal_and_save": goal_count >= 1 and save_count >= 1,
        "teacher_absent": all(not row["teacher_active"] for row in rows),
        "target_actor_executed": all(row["target_actor_active"] for row in rows),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.intended_foot_alignment_discovery.v1",
        "status": (
            "PASS_INTENDED_FOOT_ALIGNMENT_DISCOVERY"
            if all(gates.values())
            else "REJECTED_INTENDED_FOOT_ALIGNMENT_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_HOLDOUT_STRIKE_FOOT_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_adaptive_teacher_report_hash": rejected["report_hash"],
        "target_contact_actor_hash": contact.actor_hash,
        "contact_handoff_actor_hash": handoff.actor_hash,
        "body_hash": qualification.body_hash,
        "metrics": {
            "probe_count": len(rows),
            "safe_count": safe_count,
            "intended_foot_contact_count": intended_count,
            "intended_context_count": len(intended_contexts),
            "strict_chain_success_count": strict_success,
            "strict_goal_count": goal_count,
            "strict_save_count": save_count,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "activation_ceiling": "SIM_ONLY",
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
        IntendedFootAlignmentProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_dir, contact_path, handoff_path, output, index, probe, quality = job
    lead, _ = _load_lead_policy(source_dir)
    handoff = load_contact_handoff_actor(handoff_path).decide(
        contact_policy_frame=probe.action.contact_policy_frame
    )
    if not handoff.accepted or handoff.handoff_policy_frame is None:
        raise RuntimeError("contact handoff rejected intended-foot probe")
    action = probe.action
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=probe.context,
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
        shooter_ballistic_contact_torque_config=replace(
            UpperCornerStrikePolicy().torque_config(),
            contact_policy_frame=action.contact_policy_frame,
        ),
        shooter_target_velocity_contact_actor_path=contact_path,
        shooter_target_foot_velocity_xyz_mps=action.target_foot_velocity_xyz_mps,
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff.handoff_policy_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    target_active = np.asarray(
        trajectory["shooter_target_velocity_contact_actor_active"], dtype=np.bool_
    )
    teacher_active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe_hash": probe.probe_hash,
        "context": asdict(probe.context),
        "context_hash": probe.context.context_hash,
        "action": asdict(action),
        "handoff_decision": asdict(handoff),
        "result": result.to_dict(),
        "quality": strict_intended_contact_quality(
            result=result,
            trajectory=trajectory,
            quality_config=quality,
            intended_contact_foot=probe.intended_contact_foot,
        ),
        "target_actor_active": bool(np.any(target_active)),
        "teacher_active": bool(np.any(teacher_active)),
        "trajectory": artifact,
    }


def _load_rejected_teacher(path: Path) -> dict[str, Any]:
    report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        or report.get("promotion_eligible") is not False
        or len(report.get("rows", ())) < 24
    ):
        raise ValueError("intended-foot discovery requires intact rejected teacher evidence")
    for row in report["rows"]:
        artifact = row["trajectory"]
        trajectory_path = path.parent / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("rejected teacher trajectory changed")
    report["report_hash"] = claimed
    return report


def _implementation_hash() -> str:
    paths = (Path(__file__), Path(__file__).parents[1] / "growth" / "contact_handoff_actor.py")
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("intended-foot alignment output must be new and external")
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
    "IntendedFootAlignmentProbe",
    "default_intended_foot_alignment_probes",
    "run_intended_foot_alignment_discovery",
    "strict_intended_contact_quality",
]
