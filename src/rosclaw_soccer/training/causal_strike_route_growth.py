"""Auditable discovery and training for failure-aware causal strike routing."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
from rosclaw_soccer.growth.causal_strike_router import (
    CausalStrikeRouteAction,
    CausalStrikeRouteMemory,
    G1FailureAwareCausalStrikeRouter,
    save_causal_strike_router,
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


def causal_strike_route_features(context: CausalTransitionContext) -> tuple[float, ...]:
    return (
        context.receiver_lane_m,
        context.reception_target_x_m,
        context.passer_ball_local_xy_m[0],
        context.passer_ball_local_xy_m[1],
        context.ball_ground_friction,
    )


def default_causal_strike_route_actions() -> tuple[CausalStrikeRouteAction, ...]:
    return (
        CausalStrikeRouteAction(maximum_arrival_advance_frames=0),
        CausalStrikeRouteAction(maximum_arrival_advance_frames=12),
    )


def run_causal_strike_route_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    output_dir: Path,
    contexts: tuple[CausalTransitionContext, ...],
    actions: tuple[CausalStrikeRouteAction, ...] | None = None,
    option_config: G1CausalStrikeOptionConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Evaluate every discrete contact mode and retain successes and failures."""

    if len(contexts) != 6 or len({item.context_hash for item in contexts}) != 6:
        raise ValueError("causal strike route discovery requires six unique contexts")
    route_actions = actions or default_causal_strike_route_actions()
    if len(route_actions) < 2 or len(set(route_actions)) != len(route_actions):
        raise ValueError("causal strike route discovery actions must be unique")
    if not 1 <= workers <= 8:
        raise ValueError("causal strike route discovery workers must be in [1, 8]")
    output = _new_external_output(output_dir)
    quality = quality_config or CausalTransitionGrowthConfig()
    option = option_config or G1CausalStrikeOptionConfig()
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    _, source = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_route_discovery_request.v1",
        "contexts": [asdict(item) for item in contexts],
        "context_hashes": [item.context_hash for item in contexts],
        "actions": [asdict(item) for item in route_actions],
        "base_option_config": asdict(option),
        "base_option_config_hash": option.config_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "source_s95_evidence_hash": source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_training": False,
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            output,
            case_index,
            action_index,
            context,
            action,
            option,
            quality,
        )
        for case_index, context in enumerate(contexts)
        for action_index, action in enumerate(route_actions)
    )
    if workers == 1:
        rows = [_run_discovery_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_discovery_case, jobs))
    safe_count = sum(bool(row["quality"]["safe"]) for row in rows)
    success_count = sum(bool(row["quality"]["chain_passed"]) for row in rows)
    successful_contexts = len(
        {row["context_hash"] for row in rows if row["quality"]["chain_passed"]}
    )
    gates = {
        "all_candidate_worlds_safe": safe_count == len(rows),
        "minimum_successful_modes": success_count >= 4,
        "minimum_successful_contexts": successful_contexts >= 4,
        "both_actions_explored": {row["action_index"] for row in rows}
        == set(range(len(route_actions))),
        "success_and_failure_memory": 4 <= success_count < len(rows) - 3,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_route_discovery.v1",
        "status": "PASS_ROUTE_DISCOVERY" if passed else "REJECTED_ROUTE_DISCOVERY",
        "training_eligible": passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "metrics": {
            "rollout_count": len(rows),
            "safe_count": safe_count,
            "successful_mode_count": success_count,
            "successful_context_count": successful_contexts,
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "pixels_used_for_training": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "discovery-report.json", report)
    return report


def train_causal_strike_router(
    *, discovery_report_path: Path, output_dir: Path
) -> tuple[G1FailureAwareCausalStrikeRouter, dict[str, Any]]:
    source = _validate_discovery(discovery_report_path)
    output = _new_external_output(output_dir)
    rows = source["rows"]
    vectors = np.asarray([row["features"] for row in rows], dtype=np.float64)
    center = np.mean(vectors, axis=0)
    scale = np.maximum(np.ptp(vectors, axis=0), 1.0e-4)

    def memory(row: dict[str, Any]) -> CausalStrikeRouteMemory:
        return CausalStrikeRouteMemory(
            context_hash=row["context_hash"],
            trajectory_hash=row["trajectory"]["file_hash"],
            features=tuple(float(value) for value in row["features"]),
            action=CausalStrikeRouteAction(**row["action"]),
        )

    successful = tuple(memory(row) for row in rows if row["quality"]["chain_passed"])
    failed = tuple(memory(row) for row in rows if not row["quality"]["chain_passed"])
    snapshot = {
        "source_discovery_hash": source["report_hash"],
        "rows": [
            {
                "context_hash": row["context_hash"],
                "action": row["action"],
                "trajectory_hash": row["trajectory"]["file_hash"],
                "safe": row["quality"]["safe"],
                "chain_passed": row["quality"]["chain_passed"],
            }
            for row in rows
        ],
    }
    actor = G1FailureAwareCausalStrikeRouter(
        body_hash=source["body_hash"],
        kick_prior_hash=source["kick_prior_hash"],
        source_discovery_hash=source["report_hash"],
        training_snapshot_hash=hash_json(snapshot),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=successful,
        failed_memories=failed,
    )
    actor_path = output / "causal-strike-router.json"
    save_causal_strike_router(actor, actor_path)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_route_training.v1",
        "status": "PASS_FAILURE_AWARE_ROUTE_TRAINING",
        "source_discovery_hash": source["report_hash"],
        "training_snapshot_hash": actor.training_snapshot_hash,
        "successful_memory_count": len(successful),
        "failed_memory_count": len(failed),
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "algorithm": "failure_aware_nearest_verified_contact_mode",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return actor, report


def _run_discovery_case(
    job: tuple[
        Path,
        Path,
        Path,
        int,
        int,
        CausalTransitionContext,
        CausalStrikeRouteAction,
        G1CausalStrikeOptionConfig,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    (
        asset_root,
        source_s95_dir,
        output,
        case_index,
        action_index,
        context,
        action,
        option,
        quality,
    ) = job
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    kwargs.update(
        shooter_causal_strike_option_config=replace(
            option,
            maximum_arrival_advance_frames=action.maximum_arrival_advance_frames,
        ),
        shooter_ballistic_contact_torque_config=UpperCornerStrikePolicy().torque_config(),
        shooter_precontact_joint_guard_enabled=action.shooter_precontact_joint_guard,
        shooter_parameter_overrides={
            "foot_yaw_offset": action.foot_yaw_offset_rad,
            "foot_pitch_offset": 0.010,
        },
    )
    result, trajectory = simulate_shared_world(asset_root, **kwargs)
    case_dir = output / f"case-{case_index:03d}-action-{action_index:02d}"
    case_dir.mkdir(parents=True)
    artifact = _save_trajectory(case_dir / "trajectory.npz", trajectory)
    return {
        "case_index": case_index,
        "action_index": action_index,
        "case_id": context.case_id,
        "context_hash": context.context_hash,
        "features": list(causal_strike_route_features(context)),
        "action": asdict(action),
        "result": result.to_dict(),
        "quality": _chain_quality(result, trajectory, quality),
        "trajectory": artifact,
    }


def _validate_discovery(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal strike route discovery must be an object")
    claimed = payload.pop("report_hash", None)
    try:
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != "rosclaw.growth.causal_strike_route_discovery.v1"
            or payload.get("status") != "PASS_ROUTE_DISCOVERY"
            or payload.get("training_eligible") is not True
            or payload.get("implementation_hash") != _implementation_hash()
            or not all(payload.get("gates", {}).values())
        ):
            raise ValueError("causal strike route discovery authority is invalid")
        request = source.parent / "request.json"
        if not request.is_file() or hash_bytes(request.read_bytes()) != payload["request_hash"]:
            raise ValueError("causal strike route discovery request binding changed")
        for row in payload["rows"]:
            case_dir = source.parent / (
                f"case-{int(row['case_index']):03d}-action-{int(row['action_index']):02d}"
            )
            artifact = case_dir / row["trajectory"]["file"]
            if (
                not artifact.is_file()
                or hash_bytes(artifact.read_bytes()) != row["trajectory"]["file_hash"]
            ):
                raise ValueError("causal strike route trajectory binding changed")
    finally:
        if claimed is not None:
            payload["report_hash"] = claimed
    return payload


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "causal_strike_router.py",
        Path(__file__).parents[1] / "growth" / "causal_strike_option.py",
        Path(__file__).parents[1] / "growth" / "upper_corner_strike.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parent / "causal_transition_growth.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("causal strike route output must be new and external")
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
    "causal_strike_route_features",
    "default_causal_strike_route_actions",
    "run_causal_strike_route_discovery",
    "train_causal_strike_router",
]
