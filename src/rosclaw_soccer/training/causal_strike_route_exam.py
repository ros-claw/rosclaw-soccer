"""Sealed matched exam for the failure-aware causal strike router."""

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
from rosclaw_soccer.growth.causal_strike_router import load_causal_strike_router
from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.causal_strike_route_growth import causal_strike_route_features
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _chain_quality,
    _context_kwargs,
    _load_lead_policy,
    _save_trajectory,
)


def default_causal_strike_route_holdouts() -> tuple[CausalTransitionContext, ...]:
    """Fresh S126 v2 contexts registered before their first physics rollout."""

    origin = (5.10, -0.16406006503921598, 0.0)
    return (
        CausalTransitionContext(
            "s126.holdout.v2.00", origin, -0.050, 1.326, (1.201, -0.157), 0.80, 0.0990
        ),
        CausalTransitionContext(
            "s126.holdout.v2.01", origin, -0.082, 1.300, (1.209, -0.165), 0.80, 0.1020
        ),
        CausalTransitionContext(
            "s126.holdout.v2.02", origin, 0.020, 1.273, (1.197, -0.152), 0.80, 0.0915
        ),
        CausalTransitionContext(
            "s126.holdout.v2.03", origin, 0.043, 1.260, (1.215, -0.170), 0.80, 0.1070
        ),
        CausalTransitionContext(
            "s126.holdout.v2.04", origin, 0.069, 1.242, (1.200, -0.158), 0.80, 0.0950
        ),
        CausalTransitionContext(
            "s126.holdout.v2.05", origin, 0.090, 1.222, (1.220, -0.174), 0.80, 0.1050
        ),
    )


def run_causal_strike_route_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    router_path: Path,
    output_dir: Path,
    contexts: tuple[CausalTransitionContext, ...] | None = None,
    option_config: G1CausalStrikeOptionConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    sealed: bool = True,
    workers: int = 4,
) -> dict[str, Any]:
    cases = contexts or default_causal_strike_route_holdouts()
    if len(cases) != 6 or len({item.context_hash for item in cases}) != 6:
        raise ValueError("causal strike route exam requires six unique contexts")
    if not 1 <= workers <= 6:
        raise ValueError("causal strike route exam workers must be in [1, 6]")
    output = _new_external_output(output_dir)
    option = option_config or G1CausalStrikeOptionConfig()
    quality = quality_config or CausalTransitionGrowthConfig()
    router = load_causal_strike_router(router_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        router.body_hash != qualification.body_hash
        or router.kick_prior_hash != qualification.kick_prior_hash
    ):
        raise ValueError("causal strike router asset identity changed")
    _, source = _load_lead_policy(source_s95_dir)
    decisions = [router.decide(causal_strike_route_features(context)) for context in cases]
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_route_exam_request.v1",
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "sealed": sealed,
        "contexts": [asdict(item) for item in cases],
        "context_hashes": [item.context_hash for item in cases],
        "router_file_hash": hash_bytes(router_path.expanduser().resolve().read_bytes()),
        "router_hash": router.actor_hash,
        "pre_rollout_decisions": [asdict(item) for item in decisions],
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
        "pixels_used_for_scoring": False,
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            router_path.expanduser().resolve(),
            output,
            index,
            context,
            option,
            quality,
        )
        for index, context in enumerate(cases)
    )
    if workers == 1:
        rows = [_run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, jobs))
    count = len(rows)
    candidate_success = sum(bool(row["candidate_chain_passed"]) for row in rows)
    parent_success = sum(bool(row["parent_chain_passed"]) for row in rows)
    accepted_count = sum(bool(row["decision"]["accepted"]) for row in rows)
    candidate_goal_count = sum(bool(row["candidate"]["result"]["goal_crossed"]) for row in rows)
    candidate_save_count = sum(
        bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in rows
    )
    metrics = {
        "case_count": count,
        "accepted_route_count": accepted_count,
        "fallback_route_count": count - accepted_count,
        "candidate_chain_success_count": candidate_success,
        "parent_chain_success_count": parent_success,
        "candidate_chain_success_rate": candidate_success / count,
        "parent_chain_success_rate": parent_success / count,
        "chain_success_gain": candidate_success - parent_success,
        "candidate_safe_rate": sum(bool(row["candidate_safe"]) for row in rows) / count,
        "parent_safe_rate": sum(bool(row["parent_safe"]) for row in rows) / count,
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in rows) / count,
        "candidate_goal_count": candidate_goal_count,
        "candidate_save_count": candidate_save_count,
        "mean_candidate_shot_speed_mps": float(
            np.mean([row["candidate"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "mean_parent_shot_speed_mps": float(
            np.mean([row["parent"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "maximum_candidate_root_step_m": max(
            float(row["candidate"]["quality"]["maximum_root_step_m"]) for row in rows
        ),
        "maximum_candidate_ball_step_m": max(
            float(row["candidate"]["quality"]["maximum_ball_step_m"]) for row in rows
        ),
    }
    gates = {
        "router_accepts_supported_majority": accepted_count >= 4,
        "router_keeps_hard_fallbacks": count - accepted_count >= 1,
        "candidate_safe_rate": metrics["candidate_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "candidate_success_rate": candidate_success / count + 1.0e-12
        >= quality.minimum_actor_success_rate,
        "candidate_not_worse_than_parent": candidate_success >= parent_success,
        "candidate_has_measured_gain": candidate_success
        >= parent_success + quality.minimum_success_gain_cases,
        "both_goal_and_save_outcomes": candidate_goal_count >= 1 and candidate_save_count >= 1,
        "accepted_routes_realized": all(
            (not row["decision"]["accepted"])
            or (
                row["candidate"]["result"]["shooter_causal_strike_bridge_started"]
                and row["candidate_contact_torque_active_fraction"] > 0.0
            )
            for row in rows
        ),
        "safe_failure_fallbacks": all(
            row["candidate_safe"] for row in rows if not row["decision"]["accepted"]
        ),
        "terminal_option_state": all(
            row["candidate"]["result"]["shooter_causal_strike_option_final_phase"]
            in {"RECOVER", "ABORTED"}
            for row in rows
        ),
        "continuous_root_state": metrics["maximum_candidate_root_step_m"]
        <= quality.maximum_root_step_m,
        "continuous_ball_state": metrics["maximum_candidate_ball_step_m"]
        <= quality.maximum_ball_step_m,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_route_exam.v1",
        "status": (
            "PASS_CAUSAL_STRIKE_ROUTE_RETENTION"
            if sealed and passed
            else "PASS_CAUSAL_STRIKE_ROUTE_DEVELOPMENT"
            if passed
            else "REJECTED_CAUSAL_STRIKE_ROUTE"
        ),
        "sealed": sealed,
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "promotion_eligible": sealed and passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "router_hash": router.actor_hash,
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "one_shared_solver_and_ball": True,
            "world_reset_after_pass_or_shot": False,
            "pose_or_ball_teleport_after_start": False,
            "pass_and_shot_from_measured_foot_contacts": True,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
    return report


def validate_causal_strike_route_retention(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal strike route retention report must be an object")
    claimed = payload.pop("report_hash", None)
    try:
        gates = payload.get("gates")
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != "rosclaw.growth.causal_strike_route_exam.v1"
            or payload.get("status") != "PASS_CAUSAL_STRIKE_ROUTE_RETENTION"
            or payload.get("sealed") is not True
            or payload.get("promotion_eligible") is not True
            or not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
            or payload.get("implementation_hash") != _implementation_hash()
        ):
            raise ValueError("causal strike route retention authority is invalid")
        request = source.parent / "request.json"
        if not request.is_file() or hash_bytes(request.read_bytes()) != payload["request_hash"]:
            raise ValueError("causal strike route request binding changed")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError("causal strike route rows are invalid")
        for index, row in enumerate(rows):
            case_dir = source.parent / f"case-{index:03d}"
            for key in ("candidate_artifact", "replay_artifact", "parent_artifact"):
                artifact = row[key]
                artifact_path = case_dir / artifact["file"]
                if (
                    not artifact_path.is_file()
                    or hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]
                ):
                    raise ValueError("causal strike route trajectory binding changed")
    finally:
        if claimed is not None:
            payload["report_hash"] = claimed
    return payload


def _run_case(
    job: tuple[
        Path,
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        G1CausalStrikeOptionConfig,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, router_path, output, index, context, option, quality = job
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    router = load_causal_strike_router(router_path)
    decision = router.decide(causal_strike_route_features(context))
    action = decision.action
    candidate_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    candidate_kwargs.update(
        shooter_causal_strike_option_config=replace(
            option,
            maximum_arrival_advance_frames=(
                0 if action is None else action.maximum_arrival_advance_frames
            ),
        ),
        shooter_precontact_joint_guard_enabled=True,
    )
    if action is not None:
        candidate_kwargs.update(
            shooter_ballistic_contact_torque_config=UpperCornerStrikePolicy().torque_config(),
            shooter_parameter_overrides={
                "foot_yaw_offset": action.foot_yaw_offset_rad,
                "foot_pitch_offset": 0.010,
            },
        )
    parent_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    parent_kwargs["shooter_precontact_joint_guard_enabled"] = True
    candidate_result, candidate_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **candidate_kwargs)
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    candidate_artifact = _save_trajectory(case_dir / "candidate-primary.npz", candidate_trajectory)
    replay_artifact = _save_trajectory(case_dir / "candidate-replay.npz", replay_trajectory)
    parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trajectory)
    candidate_quality = _chain_quality(candidate_result, candidate_trajectory, quality)
    parent_quality = _chain_quality(parent_result, parent_trajectory, quality)
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "features": list(causal_strike_route_features(context)),
        "decision": asdict(decision),
        "candidate": {"result": candidate_result.to_dict(), "quality": candidate_quality},
        "parent": {"result": parent_result.to_dict(), "quality": parent_quality},
        "candidate_chain_passed": candidate_quality["chain_passed"],
        "parent_chain_passed": parent_quality["chain_passed"],
        "candidate_safe": candidate_quality["safe"],
        "parent_safe": parent_quality["safe"],
        "exact_replay": bool(
            candidate_result.to_dict() == replay_result.to_dict()
            and candidate_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "candidate_contact_torque_active_fraction": float(
            np.mean(candidate_trajectory["shooter_ballistic_contact_torque_active"])
        ),
        "candidate_artifact": candidate_artifact,
        "replay_artifact": replay_artifact,
        "parent_artifact": parent_artifact,
    }


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
        raise ValueError("causal strike route exam output must be new and external")
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
    "default_causal_strike_route_holdouts",
    "run_causal_strike_route_exam",
    "validate_causal_strike_route_retention",
]
