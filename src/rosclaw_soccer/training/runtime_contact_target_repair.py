"""Failure-driven target probes on a consumed runtime-target holdout."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    G1RuntimeContactTargetActor,
    RuntimeContactTargetAction,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
    _save_trajectory,
)
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    strict_intended_contact_quality,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.runtime_contact_target_exam import _case_kwargs
from rosclaw_soccer.training.runtime_receive_growth import extract_runtime_receive_features


@dataclass(frozen=True)
class RuntimeContactTargetRepairProbe:
    context: CausalTransitionContext
    playmaker_action: PlaymakerPassProbeAction
    action: RuntimeContactTargetAction

    @property
    def probe_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_runtime_contact_target_repair_probes(
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...],
) -> tuple[RuntimeContactTargetRepairProbe, ...]:
    if len(cases) != 6:
        raise ValueError("runtime contact target repair needs six consumed failures")
    targets = (
        (10.0, 3.0, -1.0),
        (9.0, 4.0, -1.0),
        (9.0, 5.0, -1.0),
        (9.0, 6.0, -1.0),
    )
    return tuple(
        RuntimeContactTargetRepairProbe(
            context=context,
            playmaker_action=playmaker_action,
            action=RuntimeContactTargetAction(target),
        )
        for context, playmaker_action in cases
        for target in targets
    )


def run_runtime_contact_target_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_exam_report_path: Path,
    target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    probes: tuple[RuntimeContactTargetRepairProbe, ...] | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("runtime contact target repair workers must be in [1, 4]")
    rejected_path = rejected_exam_report_path.expanduser().resolve()
    rejected = _bound_rejected_exam(rejected_path)
    if (
        rejected.get("status") != "REJECTED_RUNTIME_CONTACT_TARGET"
        or rejected.get("sealed") is not True
        or rejected.get("promotion_eligible") is not False
    ):
        raise ValueError("runtime contact target repair requires a rejected sealed exam")
    cases = tuple(
        (
            _context_from_dict(cast(dict[str, Any], row["context"])),
            PlaymakerPassProbeAction(**cast(dict[str, Any], row["playmaker_action"])),
        )
        for row in rejected["rows"]
    )
    active_probes = probes or default_runtime_contact_target_repair_probes(cases)
    if (
        len(active_probes) != 24
        or len({probe.probe_hash for probe in active_probes}) != 24
        or {probe.context.context_hash for probe in active_probes}
        != {context.context_hash for context, _ in cases}
    ):
        raise ValueError("runtime contact target repair curriculum is invalid")
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
        or any(
            not neural_actor.target_supported(probe.action.target_foot_velocity_xyz_mps)
            for probe in active_probes
        )
    ):
        raise ValueError("runtime contact target repair identity changed")
    handoff_decision = handoff.decide(
        contact_policy_frame=target_actor.required_receive_action.contact_policy_frame
    )
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("runtime contact target repair handoff is invalid")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    _, lead_source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_contact_target_repair_request.v1",
        "partition": "CONSUMED_REJECTED_S142_HOLDOUT",
        "probe_hashes": [probe.probe_hash for probe in active_probes],
        "rejected_exam_report_hash": rejected["report_hash"],
        "rejected_exam_file_hash": hash_bytes(rejected_path.read_bytes()),
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
        "target_intervention_clock": "CONTACT_WINDOW_AFTER_STABLE_INCOMING_OBSERVATION",
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
            target_actor,
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
    feature_by_context: dict[str, tuple[float, ...]] = {}
    for row in rows:
        features = tuple(float(value) for value in row["pre_action_features"])
        previous = feature_by_context.setdefault(str(row["context_hash"]), features)
        if not np.allclose(previous, features, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("repair target contaminated pre-action observations")
    strict_count = sum(bool(row["quality"]["strict_chain_passed"]) for row in rows)
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    recovered_contexts = {
        row["context_hash"] for row in rows if row["quality"]["strict_chain_passed"]
    }
    goals = sum(
        bool(row["quality"]["strict_chain_passed"] and row["result"]["goal_crossed"])
        for row in rows
    )
    saves = sum(
        bool(row["quality"]["strict_chain_passed"] and row["result"]["goalkeeper_save_observed"])
        for row in rows
    )
    gates = {
        "minimum_twenty_safe": safe_count >= 20,
        "minimum_four_strict": strict_count >= 4,
        "minimum_four_contexts_recovered": len(recovered_contexts) >= 4,
        "both_goal_and_save": goals >= 1 and saves >= 1,
        "pre_action_features_equal_within_context": len(feature_by_context) == 6,
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
        "schema_version": "rosclaw_soccer.runtime_contact_target_repair.v1",
        "status": (
            "PASS_RUNTIME_CONTACT_TARGET_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_RUNTIME_CONTACT_TARGET_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_REJECTED_S142_HOLDOUT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_exam_report_hash": rejected["report_hash"],
        "target_actor_hash": target_actor.actor_hash,
        "neural_actor_hash": neural_actor.actor_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": target_actor.roster_hash,
        "finisher_self_model_hash": target_actor.finisher_self_model_hash,
        "metrics": {
            "probe_count": len(rows),
            "safe_count": safe_count,
            "strict_success_count": strict_count,
            "recovered_context_count": len(recovered_contexts),
            "strict_goal_count": goals,
            "strict_save_count": saves,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "causal_contract": {
            "target_effect_starts_after_pre_action_observation": True,
            "same_pre_action_features_across_targets": True,
        },
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "repair-report.json", report)
    return report


def _run_probe(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        RuntimeContactTargetRepairProbe,
        G1RuntimeContactTargetActor,
        int,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        neural_actor_path,
        output,
        index,
        probe,
        target_actor,
        handoff_frame,
        quality,
    ) = job
    lead, _ = _load_lead_policy(source_s95_dir)
    kwargs = _case_kwargs(
        lead=lead,
        quality=quality,
        context=probe.context,
        playmaker_action=probe.playmaker_action,
        contact_actor_path=neural_actor_path,
        target_actor=target_actor,
        target_actor_path=None,
        target_velocity=probe.action.target_foot_velocity_xyz_mps,
        handoff_frame=handoff_frame,
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
        "action": asdict(probe.action),
        "pre_action_features": list(extract_runtime_receive_features(output / relative)),
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


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "runtime_contact_target_actor.py",
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parent / "runtime_contact_target_exam.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _bound_rejected_exam(path: Path) -> dict[str, Any]:
    report = _bound_json(path)
    request_path = path.parent / "request.json"
    rows = report.get("rows")
    if (
        report.get("schema_version") != "rosclaw_soccer.runtime_contact_target_exam.v1"
        or report.get("status") != "REJECTED_RUNTIME_CONTACT_TARGET"
        or report.get("sealed") is not True
        or report.get("promotion_eligible") is not False
        or report.get("evidence_boundary", {}).get("physics_authority") != "CPU_MUJOCO"
        or report.get("evidence_boundary", {}).get("hardware_command_sent") is not False
        or not request_path.is_file()
        or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or not isinstance(rows, list)
        or len(rows) != 6
    ):
        raise ValueError("rejected runtime contact target evidence is invalid")
    for index, row in enumerate(cast(list[dict[str, Any]], rows)):
        case_dir = path.parent / f"case-{index:03d}"
        for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
            artifact = cast(dict[str, Any], row[key])
            artifact_path = case_dir / str(artifact["file"])
            if hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("rejected runtime contact target trajectory changed")
    return report


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("runtime contact target repair output must be new and external")
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
    "RuntimeContactTargetRepairProbe",
    "default_runtime_contact_target_repair_probes",
    "run_runtime_contact_target_repair",
]
