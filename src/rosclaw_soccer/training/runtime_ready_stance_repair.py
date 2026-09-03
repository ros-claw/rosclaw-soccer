"""Select one universal pre-action RECEIVE stance across consumed failures."""

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
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import load_runtime_contact_target_actor
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
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


@dataclass(frozen=True)
class RuntimeReadyStanceProbe:
    context: CausalTransitionContext
    playmaker_action: PlaymakerPassProbeAction
    stance_offset_y_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.stance_offset_y_m) or self.stance_offset_y_m not in {
            -0.12,
            -0.09,
            -0.06,
            -0.03,
        }:
            raise ValueError("runtime ready stance probe is outside its preregistered set")

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_runtime_ready_stance_probes(
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...],
) -> tuple[RuntimeReadyStanceProbe, ...]:
    if len(cases) != 5:
        raise ValueError("runtime ready stance repair needs five unresolved contexts")
    return tuple(
        RuntimeReadyStanceProbe(context, playmaker, stance)
        for context, playmaker in cases
        for stance in (-0.12, -0.09, -0.06, -0.03)
    )


def run_runtime_ready_stance_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_phase_repair_path: Path,
    target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    probes: tuple[RuntimeReadyStanceProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("runtime ready stance repair workers must be in [1, 4]")
    rejected_path = rejected_phase_repair_path.expanduser().resolve()
    rejected = _bound_rejected_phase_repair(rejected_path)
    cases_by_hash: dict[str, tuple[CausalTransitionContext, PlaymakerPassProbeAction]] = {}
    for row in rejected["rows"]:
        context = _context_from_dict(row["context"])
        cases_by_hash.setdefault(
            context.context_hash,
            (context, PlaymakerPassProbeAction(**row["playmaker_action"])),
        )
    cases = tuple(cases_by_hash.values())
    active_probes = probes or default_runtime_ready_stance_probes(cases)
    if len(active_probes) != 20 or len({probe.probe_hash for probe in active_probes}) != 20:
        raise ValueError("runtime ready stance curriculum is invalid")
    target_path = target_actor_path.expanduser().resolve()
    target_actor = load_runtime_contact_target_actor(target_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural_actor = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        rejected.get("target_actor_hash") != target_actor.actor_hash
        or rejected.get("neural_actor_hash") != neural_actor.actor_hash
        or target_actor.neural_contact_actor_hash != neural_actor.actor_hash
        or target_actor.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
        or not neural_actor.target_supported((9.0, 5.0, -1.0))
    ):
        raise ValueError("runtime ready stance repair identity changed")
    contact_frame = 248
    handoff_decision = handoff.decide(contact_policy_frame=contact_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime ready stance handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_ready_stance_repair_request.v1",
        "partition": "CONSUMED_S144_PHASE_FAILURES",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "rejected_phase_repair_hash": rejected["report_hash"],
        "rejected_phase_repair_file_hash": hash_bytes(rejected_path.read_bytes()),
        "target_actor_hash": target_actor.actor_hash,
        "target_actor_file_hash": hash_bytes(target_path.read_bytes()),
        "neural_actor_hash": neural_actor.actor_hash,
        "neural_actor_file_hash": hash_bytes(neural_path.read_bytes()),
        "handoff_actor_hash": handoff.actor_hash,
        "handoff_actor_file_hash": hash_bytes(handoff_path.read_bytes()),
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "implementation_hash": _implementation_hash(),
        "foundation_search_contract": {
            "one_universal_stance_selected_across_contexts": True,
            "stance_is_not_predicted_from_post_treatment_features": True,
            "contact_policy_frame": contact_frame,
            "target_foot_velocity_xyz_mps": [9.0, 5.0, -1.0],
        },
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
            neural_path,
            output,
            index,
            probe,
            handoff_decision.handoff_policy_frame,
            quality,
        )
        for index, probe in enumerate(active_probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    summaries = [_stance_summary(rows, stance) for stance in (-0.12, -0.09, -0.06, -0.03)]
    selected = min(
        summaries,
        key=lambda item: (
            -item["strict_success_count"],
            -item["intended_foot_count"],
            -item["clear_outcome_count"],
            -item["safe_count"],
            -item["mean_shot_speed_mps"],
            item["stance_offset_y_m"],
        ),
    )
    gates = {
        "selected_safe_all_five": selected["safe_count"] == 5,
        "selected_strict_at_least_three": selected["strict_success_count"] >= 3,
        "selected_intended_foot_at_least_three": selected["intended_foot_count"] >= 3,
        "single_foundation_selected": len(summaries) == 4,
        "neural_actor_executed_all": all(row["actor_active"] for row in rows),
        "teacher_and_scripted_contact_absent": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_ready_stance_repair.v1",
        "status": (
            "PASS_RUNTIME_READY_STANCE_FOUNDATION"
            if all(gates.values())
            else "REJECTED_RUNTIME_READY_STANCE_FOUNDATION"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_S144_PHASE_FAILURES",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_phase_repair_hash": rejected["report_hash"],
        "target_actor_hash": target_actor.actor_hash,
        "neural_actor_hash": neural_actor.actor_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "stance_summaries": summaries,
        "selected_foundation": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "foundation_search_contract": request["foundation_search_contract"],
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "foundation-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        RuntimeReadyStanceProbe,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_dir, neural_path, output, index, probe, handoff_frame, quality = job
    lead, _ = _load_lead_policy(source_dir)
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
    kwargs["passer_yaw_rad"] = math.atan2(
        math.sin(base_yaw + probe.playmaker_action.body_yaw_correction_rad),
        math.cos(base_yaw + probe.playmaker_action.body_yaw_correction_rad),
    )
    action = RuntimeReceiveAction(
        maximum_arrival_advance_frames=18,
        arrival_alignment_tolerance_sec=0.02,
        stance_offset_x_m=-0.08,
        stance_offset_y_m=probe.stance_offset_y_m,
        contact_policy_frame=248,
        foot_yaw_offset_rad=-0.04,
        foot_pitch_offset_rad=0.01,
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
            arrival_alignment_tolerance_sec=action.arrival_alignment_tolerance_sec,
        ),
        shooter_neural_contact_actor_path=neural_path,
        shooter_neural_contact_policy_frame=action.contact_policy_frame,
        shooter_neural_contact_target_velocity_xyz_mps=(9.0, 5.0, -1.0),
        shooter_precontact_joint_guard_enabled=True,
        shooter_post_policy_frame=handoff_frame,
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    relative = Path(f"probe-{index:03d}.npz")
    artifact = _save_trajectory(output / relative, trajectory)
    artifact["file"] = str(relative)
    return {
        "probe_index": index,
        "probe_hash": probe.probe_hash,
        "context": asdict(probe.context),
        "context_hash": probe.context.context_hash,
        "playmaker_action": asdict(probe.playmaker_action),
        "stance_offset_y_m": probe.stance_offset_y_m,
        "result": result.to_dict(),
        "quality": strict_intended_contact_quality(
            result=result,
            trajectory=trajectory,
            quality_config=quality,
            intended_contact_foot=1,
        ),
        "actor_active": bool(np.any(trajectory["shooter_neural_contact_actor_active"])),
        "teacher_active": bool(np.any(trajectory["shooter_loft_teacher_active"])),
        "scripted_contact_active": bool(
            np.any(trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "trajectory": artifact,
    }


def _stance_summary(rows: list[dict[str, Any]], stance: float) -> dict[str, Any]:
    selected = [row for row in rows if row["stance_offset_y_m"] == stance]
    return {
        "stance_offset_y_m": stance,
        "case_count": len(selected),
        "safe_count": sum(bool(row["quality"]["safe"]) for row in selected),
        "intended_foot_count": sum(
            bool(row["quality"]["intended_foot_contact"]) for row in selected
        ),
        "strict_success_count": sum(
            bool(row["quality"]["strict_chain_passed"]) for row in selected
        ),
        "clear_outcome_count": sum(bool(row["quality"]["clear_outcome"]) for row in selected),
        "mean_shot_speed_mps": float(
            np.mean([row["result"]["shot_peak_ball_speed_mps"] for row in selected])
        ),
    }


def _bound_rejected_phase_repair(path: Path) -> dict[str, Any]:
    report = _bound_json(path)
    request_path = path.parent / "request.json"
    rows = report.get("rows")
    if (
        report.get("schema_version") != "rosclaw_soccer.runtime_receive_strike_repair.v1"
        or report.get("status") != "REJECTED_RUNTIME_RECEIVE_STRIKE_REPAIR_DATA"
        or report.get("promotion_eligible") is not False
        or report.get("intervention_contract", {}).get("only_contact_policy_frame_varied")
        is not True
        or not request_path.is_file()
        or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or not isinstance(rows, list)
        or len(rows) != 20
    ):
        raise ValueError("rejected phase repair evidence is invalid")
    for row in cast(list[dict[str, Any]], rows):
        artifact = cast(dict[str, Any], row["trajectory"])
        if hash_bytes((path.parent / str(artifact["file"])).read_bytes()) != artifact["file_hash"]:
            raise ValueError("rejected phase repair trajectory changed")
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_receive_actor.py",
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
        raise ValueError("runtime ready stance output must be new and external")
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
    "RuntimeReadyStanceProbe",
    "default_runtime_ready_stance_probes",
    "run_runtime_ready_stance_repair",
]
