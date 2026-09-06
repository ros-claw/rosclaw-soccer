"""Measured-failure curriculum for a wider target-conditioned contact actor."""

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
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.shoot.loft_teacher import G1LoftTeacherConfig
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
class AdaptiveTargetTeacherProbe:
    context: CausalTransitionContext
    action: TargetContactPlanAction
    maximum_teacher_force_xyz_n: tuple[float, float, float] = (160.0, 120.0, 100.0)

    def __post_init__(self) -> None:
        force = np.asarray(self.maximum_teacher_force_xyz_n, dtype=np.float64)
        if (
            force.shape != (3,)
            or not np.all(np.isfinite(force))
            or np.any(force < 20.0)
            or np.any(force > 200.0)
        ):
            raise ValueError("adaptive teacher force envelope is invalid")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def teacher_config(self) -> G1LoftTeacherConfig:
        target = self.action.target_foot_velocity_xyz_mps
        force = self.maximum_teacher_force_xyz_n
        return G1LoftTeacherConfig(
            target_forward_speed_mps=target[0],
            target_lateral_speed_mps=target[1],
            target_vertical_speed_mps=target[2],
            maximum_forward_force_n=force[0],
            maximum_lateral_force_n=force[1],
            maximum_vertical_force_n=force[2],
            maximum_foot_ball_distance_m=0.50,
        )


def default_adaptive_target_teacher_probes(
    contexts: tuple[CausalTransitionContext, ...],
) -> tuple[AdaptiveTargetTeacherProbe, ...]:
    if len(contexts) != 6:
        raise ValueError("adaptive target teacher needs six consumed contexts")
    corrections = (-3.0, 3.0, -3.0, -1.0, -3.0, 3.0)
    probes: list[AdaptiveTargetTeacherProbe] = []
    for index, (context, lateral) in enumerate(zip(contexts, corrections, strict=True)):
        if index < 3:
            actions = (
                TargetContactPlanAction(0, 0.12, -0.06, 248, 0.04, 0.01, (9.0, lateral, 3.0)),
                TargetContactPlanAction(6, 0.12, -0.06, 244, 0.00, 0.01, (7.0, lateral, 3.0)),
                TargetContactPlanAction(6, 0.08, -0.06, 252, 0.04, 0.01, (9.0, lateral, -1.0)),
                TargetContactPlanAction(0, 0.04, -0.06, 248, 0.00, 0.01, (9.0, 0.0, 3.0)),
            )
        else:
            actions = (
                TargetContactPlanAction(12, -0.12, -0.06, 248, -0.04, 0.01, (9.0, lateral, 3.0)),
                TargetContactPlanAction(6, -0.12, -0.06, 244, 0.00, 0.01, (7.0, lateral, 3.0)),
                TargetContactPlanAction(18, -0.08, -0.06, 252, -0.04, 0.01, (9.0, lateral, -1.0)),
                TargetContactPlanAction(12, -0.12, -0.06, 248, 0.00, 0.01, (9.0, 0.0, 3.0)),
            )
        probes.extend(AdaptiveTargetTeacherProbe(context, action) for action in actions)
    return tuple(probes)


def run_adaptive_target_teacher_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_recovered_discovery_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    probes: tuple[AdaptiveTargetTeacherProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("adaptive target teacher workers must be in [1, 8]")
    rejected_path = rejected_recovered_discovery_path.expanduser().resolve()
    rejected = _load_rejected_recovered_discovery(rejected_path)
    unique_contexts: dict[str, CausalTransitionContext] = {}
    for row in rejected["rows"]:
        context = _context_from_dict(row["context"])
        unique_contexts.setdefault(context.context_hash, context)
    contexts = tuple(unique_contexts.values())
    active_probes = probes or default_adaptive_target_teacher_probes(contexts)
    if (
        len(active_probes) < 24
        or len({probe.probe_hash for probe in active_probes}) != len(active_probes)
        or {probe.context.context_hash for probe in active_probes} != set(unique_contexts)
    ):
        raise ValueError("adaptive target curriculum lacks consumed context diversity")
    targets = np.asarray(
        [probe.action.target_foot_velocity_xyz_mps for probe in active_probes], dtype=np.float64
    )
    if any(np.ptp(targets[:, axis]) <= 1.0e-9 for axis in range(3)):
        raise ValueError("adaptive target curriculum must vary every target axis")
    handoff = load_contact_handoff_actor(handoff_actor_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        handoff.body_hash != qualification.body_hash
        or rejected["contact_handoff_actor_hash"] != handoff.actor_hash
    ):
        raise ValueError("adaptive target teacher handoff lineage changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.adaptive_target_teacher_request.v1",
        "partition": "CONSUMED_REJECTED_CONTACT_RESPONSE_TEACHER_DEVELOPMENT",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "rejected_recovered_report_hash": rejected["report_hash"],
        "rejected_recovered_file_hash": hash_bytes(rejected_path.read_bytes()),
        "contact_handoff_actor_hash": handoff.actor_hash,
        "contact_handoff_actor_file_hash": hash_bytes(handoff_actor_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_role": "SIM_ONLY_DATA_GENERATOR",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
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
    active_count = sum(bool(row["teacher_active"]) for row in rows)
    goal_count = sum(bool(row["result"]["goal_crossed"]) for row in rows)
    save_count = sum(bool(row["result"]["goalkeeper_save_observed"]) for row in rows)
    successful_contexts = {row["context_hash"] for row in rows if row["quality"]["chain_passed"]}
    gates = {
        "safe_training_support": safe_count >= len(rows) - 4,
        "minimum_successes": success_count >= 4,
        "minimum_context_coverage": len(successful_contexts) >= 3,
        "both_support_clusters": any(row["quality"]["chain_passed"] for row in rows[:12])
        and any(row["quality"]["chain_passed"] for row in rows[12:]),
        "both_goal_and_save": goal_count >= 1 and save_count >= 1,
        "teacher_executed": active_count == len(rows),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.adaptive_target_teacher_discovery.v1",
        "status": (
            "PASS_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
            if all(gates.values())
            else "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_CONTACT_RESPONSE_TEACHER_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_recovered_report_hash": rejected["report_hash"],
        "contact_handoff_actor_hash": handoff.actor_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "metrics": {
            "probe_count": len(rows),
            "chain_success_count": success_count,
            "safe_count": safe_count,
            "successful_context_count": len(successful_contexts),
            "goal_count": goal_count,
            "save_count": save_count,
            "teacher_active_count": active_count,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_role": "SIM_ONLY_DATA_GENERATOR",
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
        int,
        AdaptiveTargetTeacherProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_dir, handoff_path, output, index, probe, quality = job
    lead, _ = _load_lead_policy(source_dir)
    handoff = load_contact_handoff_actor(handoff_path).decide(
        contact_policy_frame=probe.action.contact_policy_frame
    )
    if not handoff.accepted or handoff.handoff_policy_frame is None:
        raise RuntimeError("learned handoff rejected adaptive teacher action")
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
        shooter_loft_teacher_config=probe.teacher_config(),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff.handoff_policy_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe_hash": probe.probe_hash,
        "context": asdict(probe.context),
        "context_hash": probe.context.context_hash,
        "action": asdict(action),
        "maximum_teacher_force_xyz_n": list(probe.maximum_teacher_force_xyz_n),
        "teacher_config_hash": probe.teacher_config().config_hash,
        "handoff_decision": asdict(handoff),
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "teacher_active": bool(np.any(active)),
        "teacher_active_frame_count": int(np.count_nonzero(active)),
        "trajectory": artifact,
    }


def _load_rejected_recovered_discovery(path: Path) -> dict[str, Any]:
    report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("status") != "REJECTED_RECOVERED_TARGET_CONTACT_DISCOVERY"
        or report.get("promotion_eligible") is not False
        or len(report.get("rows", ())) < 24
    ):
        raise ValueError("adaptive teacher requires intact rejected contact responses")
    for row in report["rows"]:
        artifact = row["trajectory"]
        trajectory_path = path.parent / artifact["file"]
        if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
            raise ValueError("rejected contact response trajectory changed")
    report["report_hash"] = claimed
    return report


def _implementation_hash() -> str:
    paths = (
        Path(__file__),
        Path(__file__).parents[1] / "skills" / "shoot" / "loft_teacher.py",
        Path(__file__).parents[1] / "growth" / "contact_handoff_actor.py",
    )
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("adaptive target teacher output must be new and external")
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
    "AdaptiveTargetTeacherProbe",
    "default_adaptive_target_teacher_probes",
    "run_adaptive_target_teacher_discovery",
]
