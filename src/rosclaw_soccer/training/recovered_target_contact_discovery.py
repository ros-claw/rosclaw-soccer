"""Explore contact plans on a learned post-contact safety handoff.

This stage is deliberately downstream of a rejected sealed exam.  The old
holdouts are consumed as development data, while the learned handoff actor
keeps every counterfactual on the same recovery authority boundary.
"""

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
from rosclaw_soccer.training.contact_handoff_discovery import (
    _context_from_dict,
    _load_rejected_exam,
)


@dataclass(frozen=True)
class RecoveredTargetContactProbe:
    context: CausalTransitionContext
    action: TargetContactPlanAction

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_recovered_target_contact_probes(
    contexts: tuple[CausalTransitionContext, ...],
) -> tuple[RecoveredTargetContactProbe, ...]:
    """Register four counterfactuals per consumed S129 support cluster."""

    if len(contexts) != 6:
        raise ValueError("recovered target contact discovery needs six consumed contexts")
    first_cluster = (
        TargetContactPlanAction(0, 0.04, -0.06, 248, 0.04, 0.01, (7.0, 1.0, 3.0)),
        TargetContactPlanAction(0, 0.12, -0.06, 248, 0.04, 0.01, (5.0, 0.0, 0.0)),
        TargetContactPlanAction(0, 0.12, -0.06, 248, 0.04, 0.01, (7.0, 1.0, 3.0)),
        TargetContactPlanAction(0, 0.08, -0.06, 248, 0.04, 0.01, (5.0, 0.0, 0.0)),
    )
    second_cluster = (
        TargetContactPlanAction(12, -0.12, -0.06, 248, -0.04, 0.01, (7.0, 1.0, 3.0)),
        TargetContactPlanAction(12, -0.12, -0.06, 248, -0.04, 0.01, (5.0, 0.0, 0.0)),
        TargetContactPlanAction(12, -0.12, -0.06, 248, 0.00, 0.01, (5.0, 0.0, 0.0)),
        TargetContactPlanAction(12, -0.08, -0.06, 248, -0.04, 0.01, (7.0, 1.0, 3.0)),
    )
    return tuple(
        RecoveredTargetContactProbe(context, action)
        for index, context in enumerate(contexts)
        for action in (first_cluster if index < 3 else second_cluster)
    )


def run_recovered_target_contact_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_path: Path,
    target_contact_actor_path: Path,
    handoff_actor_path: Path,
    handoff_training_report_path: Path,
    output_dir: Path,
    probes: tuple[RecoveredTargetContactProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("recovered target contact workers must be in [1, 8]")
    rejected_path = rejected_exam_path.expanduser().resolve()
    rejected = _load_rejected_exam(rejected_path)
    contexts = tuple(_context_from_dict(row["context"]) for row in rejected["rows"])
    active_probes = probes or default_recovered_target_contact_probes(contexts)
    consumed_hashes = {context.context_hash for context in contexts}
    if (
        len(active_probes) < 24
        or len({probe.probe_hash for probe in active_probes}) != len(active_probes)
        or {probe.context.context_hash for probe in active_probes} != consumed_hashes
        or any(
            sum(probe.context.context_hash == context_hash for probe in active_probes) < 4
            for context_hash in consumed_hashes
        )
    ):
        raise ValueError("recovered target contact probes lack consumed failure diversity")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    contact = load_g1_target_velocity_contact_actor(target_contact_actor_path)
    handoff = load_contact_handoff_actor(handoff_actor_path)
    handoff_training = _bound_report(handoff_training_report_path)
    if (
        {contact.body_hash, handoff.body_hash} != {qualification.body_hash}
        or handoff.target_contact_actor_hash != contact.actor_hash
        or handoff.target_plan_actor_hash != rejected["target_plan_actor_hash"]
        or handoff_training.get("status") != "PASS_CONTACT_HANDOFF_TRAINING"
        or handoff_training.get("actor_hash") != handoff.actor_hash
        or handoff_training.get("actor_file_hash") != hash_bytes(handoff_actor_path.read_bytes())
        or tuple(handoff.source_evidence_hashes)
        != (handoff_training.get("source_discovery_report_hash"),)
    ):
        raise ValueError("recovered target contact actor lineage changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.recovered_target_contact_discovery_request.v1",
        "partition": "CONSUMED_REJECTED_HOLDOUT_SAFE_CONTACT_DEVELOPMENT",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "rejected_exam_report_hash": rejected["report_hash"],
        "rejected_exam_file_hash": hash_bytes(rejected_path.read_bytes()),
        "target_contact_actor_hash": contact.actor_hash,
        "target_contact_actor_file_hash": hash_bytes(target_contact_actor_path.read_bytes()),
        "contact_handoff_actor_hash": handoff.actor_hash,
        "contact_handoff_actor_file_hash": hash_bytes(handoff_actor_path.read_bytes()),
        "contact_handoff_training_report_hash": handoff_training["report_hash"],
        "contact_handoff_training_file_hash": hash_bytes(handoff_training_report_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "late_stance_rewrite_allowed": False,
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
    success_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    goal_count = sum(bool(row["result"]["goal_crossed"]) for row in rows)
    save_count = sum(bool(row["result"]["goalkeeper_save_observed"]) for row in rows)
    context_success_count = len(
        {row["context_hash"] for row in rows if row["quality"]["chain_passed"]}
    )
    actor_active_count = sum(bool(row["target_actor_active"]) for row in rows)
    first_cluster_success = any(row["quality"]["chain_passed"] for row in rows[:12])
    second_cluster_success = any(row["quality"]["chain_passed"] for row in rows[12:])
    gates = {
        "all_safe": safe_count == len(rows),
        "minimum_successes": success_count >= 4,
        "minimum_context_coverage": context_success_count >= 3,
        "both_support_clusters": first_cluster_success and second_cluster_success,
        "both_goal_and_save": goal_count >= 1 and save_count >= 1,
        "teacher_absent": all(not row["teacher_active"] for row in rows),
        "target_actor_executed": actor_active_count == len(rows),
        "learned_handoff_applied": all(row["handoff_decision"]["accepted"] for row in rows),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.recovered_target_contact_discovery.v1",
        "status": (
            "PASS_RECOVERED_TARGET_CONTACT_DISCOVERY"
            if all(gates.values())
            else "REJECTED_RECOVERED_TARGET_CONTACT_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_HOLDOUT_SAFE_CONTACT_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_report_hash": rejected["report_hash"],
        "target_contact_actor_hash": contact.actor_hash,
        "contact_handoff_actor_hash": handoff.actor_hash,
        "body_hash": qualification.body_hash,
        "metrics": {
            "probe_count": len(rows),
            "chain_success_count": success_count,
            "safe_count": safe_count,
            "successful_context_count": context_success_count,
            "goal_count": goal_count,
            "save_count": save_count,
            "target_actor_active_count": actor_active_count,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "late_stance_rewrite_allowed": False,
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
        RecoveredTargetContactProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_dir, contact_path, handoff_path, output, index, probe, quality = job
    lead, _ = _load_lead_policy(source_dir)
    handoff = load_contact_handoff_actor(handoff_path).decide(
        contact_policy_frame=probe.action.contact_policy_frame
    )
    if not handoff.accepted or handoff.handoff_policy_frame is None:
        raise RuntimeError("learned contact handoff rejected an in-support action")
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
        "quality": _chain_quality(result, trajectory, quality),
        "target_actor_active": bool(np.any(target_active)),
        "target_actor_active_frame_count": int(np.count_nonzero(target_active)),
        "teacher_active": bool(np.any(teacher_active)),
        "trajectory": artifact,
    }


def _bound_report(path: Path) -> dict[str, Any]:
    payload = cast(
        dict[str, Any],
        json.loads(path.expanduser().resolve().read_text(encoding="utf-8")),
    )
    claimed = payload.pop("report_hash", None)
    if claimed != hash_json(payload):
        raise ValueError("bound report integrity changed")
    payload["report_hash"] = claimed
    return payload


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "contact_handoff_actor.py",
        Path(__file__).parents[1] / "growth" / "target_velocity_contact_actor.py",
    )
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("recovered target contact output must be new and external")
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
    "RecoveredTargetContactProbe",
    "default_recovered_target_contact_probes",
    "run_recovered_target_contact_discovery",
]
