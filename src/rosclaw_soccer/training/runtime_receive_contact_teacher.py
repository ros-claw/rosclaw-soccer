"""Targeted low-level contact curriculum from a rejected runtime RECEIVE frontier."""

from __future__ import annotations

import hashlib
import json
import math
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
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction


@dataclass(frozen=True)
class RuntimeReceiveContactProbe:
    """One bounded task-space target for the frozen role-ready strike."""

    context: CausalTransitionContext
    playmaker_action: PlaymakerPassProbeAction
    action: TargetContactPlanAction
    maximum_teacher_force_xyz_n: tuple[float, float, float] = (180.0, 180.0, 100.0)

    def __post_init__(self) -> None:
        force = np.asarray(self.maximum_teacher_force_xyz_n, dtype=np.float64)
        if (
            force.shape != (3,)
            or not np.all(np.isfinite(force))
            or np.any(force < 20.0)
            or np.any(force > 200.0)
            or self.action.maximum_arrival_advance_frames != 18
        ):
            raise ValueError("runtime RECEIVE contact probe is outside its teacher envelope")

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


def default_runtime_receive_contact_probes(
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...],
) -> tuple[RuntimeReceiveContactProbe, ...]:
    """Search the unsupported lateral target axis without changing ready posture."""

    if len(cases) != 2 or len({context.context_hash for context, _ in cases}) != 2:
        raise ValueError("runtime RECEIVE contact teacher needs two unique failed contexts")
    targets = (
        (9.0, -6.0, -1.0),
        (9.0, -5.0, -1.0),
        (9.0, -4.0, -1.0),
        (9.0, -3.0, -1.0),
        (9.0, -2.0, -1.0),
        (9.0, -1.0, -1.0),
        (10.0, -6.0, -1.0),
        (11.0, -6.0, -1.0),
    )
    return tuple(
        RuntimeReceiveContactProbe(
            context=context,
            playmaker_action=playmaker_action,
            action=TargetContactPlanAction(
                maximum_arrival_advance_frames=18,
                stance_offset_x_m=-0.08,
                stance_offset_y_m=-0.06,
                contact_policy_frame=258,
                foot_yaw_offset_rad=-0.04,
                foot_pitch_offset_rad=0.01,
                target_foot_velocity_xyz_mps=target,
            ),
        )
        for context, playmaker_action in cases
        for target in targets
    )


def runtime_receive_contact_countersteer_probes(
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...],
) -> tuple[RuntimeReceiveContactProbe, ...]:
    """Follow the measured S140 lateral gradient through zero and positive targets."""

    if len(cases) != 2 or len({context.context_hash for context, _ in cases}) != 2:
        raise ValueError("runtime RECEIVE contact teacher needs two unique failed contexts")
    targets = (
        (9.0, 0.0, -1.0),
        (9.0, 1.0, -1.0),
        (9.0, 2.0, -1.0),
        (9.0, 3.0, -1.0),
        (9.0, 4.0, -1.0),
        (9.0, 5.0, -1.0),
        (9.0, 6.0, -1.0),
        (10.0, 3.0, -1.0),
    )
    return tuple(
        RuntimeReceiveContactProbe(
            context=context,
            playmaker_action=playmaker_action,
            action=TargetContactPlanAction(
                maximum_arrival_advance_frames=18,
                stance_offset_x_m=-0.08,
                stance_offset_y_m=-0.06,
                contact_policy_frame=258,
                foot_yaw_offset_rad=-0.04,
                foot_pitch_offset_rad=0.01,
                target_foot_velocity_xyz_mps=target,
            ),
        )
        for context, playmaker_action in cases
        for target in targets
    )


def run_runtime_receive_contact_teacher(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_direction_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    probes: tuple[RuntimeReceiveContactProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Generate multi-target teacher traces only after upper-layer exhaustion."""

    if not 1 <= workers <= 4:
        raise ValueError("runtime RECEIVE contact teacher workers must be in [1, 4]")
    rejected_path = rejected_direction_report_path.expanduser().resolve()
    rejected = _bound_rejected_direction(rejected_path)
    cases_by_hash: dict[str, tuple[CausalTransitionContext, PlaymakerPassProbeAction]] = {}
    for row in rejected["rows"]:
        context = CausalTransitionContext(**row["context"])
        cases_by_hash.setdefault(
            context.context_hash,
            (context, PlaymakerPassProbeAction(**row["playmaker_action"])),
        )
    cases = tuple(cases_by_hash.values())
    active_probes = probes or default_runtime_receive_contact_probes(cases)
    if (
        len(active_probes) != 16
        or len({probe.probe_hash for probe in active_probes}) != 16
        or {probe.context.context_hash for probe in active_probes}
        != {context.context_hash for context, _ in cases}
    ):
        raise ValueError("runtime RECEIVE contact curriculum is invalid")
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff_actor = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if handoff_actor.body_hash != qualification.body_hash:
        raise ValueError("runtime RECEIVE contact teacher body identity changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_contact_teacher_request.v1",
        "partition": "CONSUMED_REJECTED_RUNTIME_RECEIVE_DIRECTION_FRONTIER",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "context_hashes": [context.context_hash for context, _ in cases],
        "rejected_direction_report_hash": rejected["report_hash"],
        "rejected_direction_file_hash": hash_bytes(rejected_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "handoff_actor_hash": handoff_actor.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "teacher_role": "SIM_ONLY_LOW_LEVEL_CONTACT_DATA_GENERATOR",
        "upper_layer_exhausted": True,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            handoff_path,
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
    recovered = {row["context_hash"] for row in rows if row["quality"]["chain_passed"]}
    selected = [_select(rows, context.context_hash) for context, _ in cases]
    gates = {
        "both_failed_contexts_recovered": len(recovered) == 2,
        "minimum_four_strict_outcomes": sum(bool(row["quality"]["chain_passed"]) for row in rows)
        >= 4,
        "minimum_twelve_safe": sum(bool(row["quality"]["safe"]) for row in rows) >= 12,
        "selected_safe": all(row["quality"]["safe"] for row in selected),
        "teacher_executed_all": all(row["teacher_active"] for row in rows),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_receive_contact_teacher.v1",
        "status": (
            "PASS_RUNTIME_RECEIVE_CONTACT_TEACHER"
            if all(gates.values())
            else "REJECTED_RUNTIME_RECEIVE_CONTACT_TEACHER"
        ),
        "promotion_eligible": False,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_direction_report_hash": rejected["report_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "metrics": {
            "probe_count": len(rows),
            "safe_count": sum(bool(row["quality"]["safe"]) for row in rows),
            "chain_success_count": sum(bool(row["quality"]["chain_passed"]) for row in rows),
            "recovered_context_count": len(recovered),
            "goal_count": sum(bool(row["result"]["goal_crossed"]) for row in rows),
            "save_count": sum(bool(row["result"]["goalkeeper_save_observed"]) for row in rows),
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "teacher_role": "SIM_ONLY_LOW_LEVEL_CONTACT_DATA_GENERATOR",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "teacher-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        RuntimeReceiveContactProbe,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_dir, handoff_path, output, index, probe, quality = job
    lead, _ = _load_lead_policy(source_dir)
    handoff = load_contact_handoff_actor(handoff_path).decide(
        contact_policy_frame=probe.action.contact_policy_frame
    )
    if not handoff.accepted or handoff.handoff_policy_frame is None:
        raise RuntimeError("runtime RECEIVE contact teacher handoff was rejected")
    action = probe.action
    kwargs = _context_kwargs(
        lead_policy=lead,
        config=quality,
        context=probe.context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    base = cast(dict[str, float], kwargs["passer_parameter_overrides"])
    kwargs["passer_parameter_overrides"] = {
        **base,
        "stance_offset_x": base["stance_offset_x"] + probe.playmaker_action.stance_correction_x_m,
        "stance_offset_y": base["stance_offset_y"] + probe.playmaker_action.stance_correction_y_m,
        "swing_speed_scale": probe.playmaker_action.swing_speed_scale,
    }
    base_yaw = float(kwargs["passer_yaw_rad"])
    correction = probe.playmaker_action.body_yaw_correction_rad
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
    teacher_active = np.asarray(trajectory["shooter_loft_teacher_active"], dtype=np.bool_)
    return {
        "probe_index": index,
        "probe_hash": probe.probe_hash,
        "context": asdict(probe.context),
        "context_hash": probe.context.context_hash,
        "playmaker_action": asdict(probe.playmaker_action),
        "action": asdict(action),
        "teacher_config_hash": probe.teacher_config().config_hash,
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "teacher_active": bool(np.any(teacher_active)),
        "teacher_active_frame_count": int(np.count_nonzero(teacher_active)),
        "trajectory": artifact,
    }


def _select(rows: list[dict[str, Any]], context_hash: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["context_hash"] == context_hash]
    return min(
        candidates,
        key=lambda row: (
            not row["quality"]["chain_passed"],
            not row["quality"]["safe"],
            -float(row["result"]["shot_peak_ball_speed_mps"]),
            row["probe_hash"],
        ),
    )


def _bound_rejected_direction(path: Path) -> dict[str, Any]:
    report = _read_object(path)
    claimed = report.pop("report_hash", None)
    if (
        claimed != hash_json(report)
        or report.get("schema_version") != "rosclaw_soccer.runtime_receive_discovery.v1"
        or report.get("status") != "REJECTED_RUNTIME_RECEIVE_DISCOVERY"
        or report.get("promotion_eligible") is not False
        or report.get("gates", {}).get("runtime_intervention_observed_all") is not True
        or len(report.get("rows", ())) != 16
    ):
        raise ValueError("runtime RECEIVE direction frontier is invalid")
    report["report_hash"] = claimed
    request = path.parent / "request.json"
    if not request.is_file() or hash_bytes(request.read_bytes()) != report.get("request_hash"):
        raise ValueError("runtime RECEIVE direction request changed")
    for row in report["rows"]:
        artifact = row["trajectory"]
        if hash_bytes((path.parent / artifact["file"]).read_bytes()) != artifact["file_hash"]:
            raise ValueError("runtime RECEIVE direction trajectory changed")
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "shoot" / "loft_teacher.py",
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
        raise ValueError("runtime RECEIVE contact teacher output must be new and external")
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
    "RuntimeReceiveContactProbe",
    "default_runtime_receive_contact_probes",
    "run_runtime_receive_contact_teacher",
    "runtime_receive_contact_countersteer_probes",
]
