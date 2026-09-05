"""Failure-driven discovery of post-contact recovery handoff timing."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.planned_contact_mode_actor import planned_contact_mode_features
from rosclaw_soccer.growth.target_contact_plan_actor import load_target_contact_plan_actor
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


@dataclass(frozen=True)
class ContactHandoffProbe:
    context: CausalTransitionContext
    handoff_policy_frame: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.handoff_policy_frame, bool)
            or not 238 <= self.handoff_policy_frame <= 280
        ):
            raise ValueError("contact handoff probe exceeds its SIM-only timing envelope")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_contact_handoff_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_path: Path,
    target_plan_actor_path: Path,
    target_contact_actor_path: Path,
    output_dir: Path,
    handoff_policy_frames: tuple[int, ...] = (248, 255),
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 6,
) -> dict[str, Any]:
    if (
        len(handoff_policy_frames) < 2
        or len(set(handoff_policy_frames)) != len(handoff_policy_frames)
        or any(not 238 <= value <= 280 for value in handoff_policy_frames)
        or not 1 <= workers <= 8
    ):
        raise ValueError("contact handoff discovery configuration is invalid")
    rejected_path = rejected_exam_path.expanduser().resolve()
    rejected = _load_rejected_exam(rejected_path)
    contexts = tuple(_context_from_dict(row["context"]) for row in rejected["rows"])
    probes = tuple(
        ContactHandoffProbe(context, frame)
        for context in contexts
        for frame in handoff_policy_frames
    )
    quality = quality_config or CausalTransitionGrowthConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    plan = load_target_contact_plan_actor(target_plan_actor_path)
    contact = load_g1_target_velocity_contact_actor(target_contact_actor_path)
    if (
        {plan.body_hash, contact.body_hash} != {qualification.body_hash}
        or plan.target_contact_actor_hash != contact.actor_hash
        or rejected["target_plan_actor_hash"] != plan.actor_hash
        or rejected["target_contact_actor_hash"] != contact.actor_hash
    ):
        raise ValueError("contact handoff discovery lineage changed")
    _, source = _load_lead_policy(source_s95_dir)
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.contact_handoff_discovery_request.v1",
        "partition": "CONSUMED_REJECTED_HOLDOUT_RECOVERY_DEVELOPMENT",
        "probe_hashes": [probe.probe_hash for probe in probes],
        "handoff_policy_frames": list(handoff_policy_frames),
        "rejected_exam_report_hash": rejected["report_hash"],
        "rejected_exam_file_hash": hash_bytes(rejected_path.read_bytes()),
        "target_plan_actor_hash": plan.actor_hash,
        "target_plan_actor_file_hash": hash_bytes(target_plan_actor_path.read_bytes()),
        "target_contact_actor_hash": contact.actor_hash,
        "target_contact_actor_file_hash": hash_bytes(target_contact_actor_path.read_bytes()),
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_enabled": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            target_plan_actor_path.expanduser().resolve(),
            target_contact_actor_path.expanduser().resolve(),
            output,
            index,
            probe,
            quality,
        )
        for index, probe in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    baseline = {str(row["context_hash"]): row for row in rejected["rows"]}
    summaries: dict[str, Any] = {}
    for frame in handoff_policy_frames:
        selected = [row for row in rows if row["handoff_policy_frame"] == frame]
        recovered = sum(
            not bool(baseline[row["context_hash"]]["candidate_quality"]["safe"])
            and bool(row["quality"]["safe"])
            for row in selected
        )
        regressed = sum(
            bool(baseline[row["context_hash"]]["candidate_quality"]["safe"])
            and not bool(row["quality"]["safe"])
            for row in selected
        )
        summaries[str(frame)] = {
            "safe_count": sum(bool(row["quality"]["safe"]) for row in selected),
            "chain_success_count": sum(bool(row["quality"]["chain_passed"]) for row in selected),
            "recovered_failure_count": recovered,
            "regressed_safe_count": regressed,
            "minimum_shooter_pelvis_height_m": min(
                float(row["result"]["shooter_min_pelvis_height_m"]) for row in selected
            ),
            "mean_shooter_tail_wobble_index": float(
                np.mean([row["result"]["shooter_tail_wobble_index"] for row in selected])
            ),
        }
    baseline_unsafe = sum(not bool(row["candidate_quality"]["safe"]) for row in rejected["rows"])
    passing_frames = [
        frame
        for frame in handoff_policy_frames
        if summaries[str(frame)]["safe_count"] == len(contexts)
        and summaries[str(frame)]["recovered_failure_count"] == baseline_unsafe
        and summaries[str(frame)]["regressed_safe_count"] == 0
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.contact_handoff_discovery.v1",
        "status": (
            "PASS_CONTACT_HANDOFF_DISCOVERY"
            if passing_frames
            else "REJECTED_CONTACT_HANDOFF_DISCOVERY"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_HOLDOUT_RECOVERY_DEVELOPMENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_report_hash": rejected["report_hash"],
        "target_plan_actor_hash": plan.actor_hash,
        "target_contact_actor_hash": contact.actor_hash,
        "body_hash": qualification.body_hash,
        "baseline_unsafe_count": baseline_unsafe,
        "passing_handoff_policy_frames": passing_frames,
        "frame_summaries": summaries,
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
        Path, Path, Path, Path, Path, int, ContactHandoffProbe, CausalTransitionGrowthConfig
    ],
) -> dict[str, Any]:
    asset_root, source_dir, plan_path, contact_path, output, index, probe, quality = job
    lead, _ = _load_lead_policy(source_dir)
    plan_actor = load_target_contact_plan_actor(plan_path)
    features = planned_contact_mode_features(
        receiver_lane_m=probe.context.receiver_lane_m,
        reception_target_x_m=probe.context.reception_target_x_m,
        passer_ball_local_xy_m=probe.context.passer_ball_local_xy_m,
        ball_ground_friction=probe.context.ball_ground_friction,
    )
    decision = plan_actor.decide(features)
    if decision.action is None:
        raise RuntimeError("bound target contact plan rejected its consumed context")
    action = decision.action
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
        shooter_post_policy_frame=probe.handoff_policy_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    artifact = _save_trajectory(output / f"probe-{index:03d}.npz", trajectory)
    return {
        "probe_index": index,
        "probe_hash": probe.probe_hash,
        "context_hash": probe.context.context_hash,
        "case_id": probe.context.case_id,
        "handoff_policy_frame": probe.handoff_policy_frame,
        "plan_decision": asdict(decision),
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "trajectory": artifact,
    }


def _load_rejected_exam(path: Path) -> dict[str, Any]:
    report = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("status") != "REJECTED_TARGET_CONTACT_RETENTION"
        or report.get("promotion_eligible") is not False
        or report.get("sealed") is not True
        or len(report.get("rows", ())) != 6
    ):
        raise ValueError("contact handoff discovery requires intact rejected evidence")
    for index, row in enumerate(report["rows"]):
        case_dir = path.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = row[key]
            artifact_path = case_dir / artifact["file"]
            if hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("rejected contact trajectory changed")
    report["report_hash"] = claimed
    return report


def _context_from_dict(payload: dict[str, Any]) -> CausalTransitionContext:
    value = dict(payload)
    value["passer_origin_m"] = tuple(value["passer_origin_m"])
    value["passer_ball_local_xy_m"] = tuple(value["passer_ball_local_xy_m"])
    return CausalTransitionContext(**value)


def _implementation_hash() -> str:
    paths = (Path(__file__), Path(__file__).parents[1] / "growth" / "contact_handoff_actor.py")
    return str(hash_json({str(path): hash_bytes(path.read_bytes()) for path in paths}))


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("contact handoff discovery output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["ContactHandoffProbe", "run_contact_handoff_discovery"]
