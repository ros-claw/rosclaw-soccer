"""Matched CPU-MuJoCo exams for the delayed causal strike option."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_option import G1CausalStrikeOptionConfig
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
    default_transition_holdouts,
)


def default_causal_strike_option_development_contexts() -> tuple[CausalTransitionContext, ...]:
    """The consumed S124 v6 holdouts; development-only from S125 onward."""

    return default_transition_holdouts()


def default_causal_strike_option_holdouts() -> tuple[CausalTransitionContext, ...]:
    """Fresh S125 v1 contexts, pre-registered before their first rollout."""

    return (
        CausalTransitionContext(
            "s125.holdout.v1.00",
            (5.10, -0.16406006503921598, 0.0),
            -0.033,
            1.337,
            (1.198, -0.155),
            0.80,
            0.0965,
        ),
        CausalTransitionContext(
            "s125.holdout.v1.01",
            (5.10, -0.16406006503921598, 0.0),
            -0.068,
            1.307,
            (1.212, -0.168),
            0.80,
            0.1045,
        ),
        CausalTransitionContext(
            "s125.holdout.v1.02",
            (5.10, -0.16406006503921598, 0.0),
            0.012,
            1.279,
            (1.190, -0.148),
            0.80,
            0.0895,
        ),
        CausalTransitionContext(
            "s125.holdout.v1.03",
            (5.10, -0.16406006503921598, 0.0),
            0.047,
            1.258,
            (1.218, -0.172),
            0.80,
            0.1085,
        ),
        CausalTransitionContext(
            "s125.holdout.v1.04",
            (5.10, -0.16406006503921598, 0.0),
            0.073,
            1.239,
            (1.202, -0.162),
            0.80,
            0.0935,
        ),
        CausalTransitionContext(
            "s125.holdout.v1.05",
            (5.10, -0.16406006503921598, 0.0),
            0.095,
            1.217,
            (1.225, -0.176),
            0.80,
            0.1065,
        ),
    )


def run_causal_strike_option_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    output_dir: Path,
    contexts: tuple[CausalTransitionContext, ...],
    option_config: G1CausalStrikeOptionConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    sealed: bool,
    workers: int = 4,
) -> dict[str, Any]:
    """Run option/replay/frozen-parent in matched, continuous physical worlds."""

    if len(contexts) != 6 or len({context.context_hash for context in contexts}) != 6:
        raise ValueError("causal strike option exam requires six unique contexts")
    if not 1 <= workers <= 6:
        raise ValueError("causal strike option exam workers must be in [1, 6]")
    option = option_config or G1CausalStrikeOptionConfig()
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    _, source_lead = _load_lead_policy(source_s95_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_option_exam_request.v1",
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "sealed": sealed,
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "option_config": asdict(option),
        "option_config_hash": option.config_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "source_s95_evidence_hash": source_lead["evidence_hash"],
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
            output,
            index,
            context,
            option,
            quality,
        )
        for index, context in enumerate(contexts)
    )
    if workers == 1:
        rows = [_run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_case, jobs))
    count = len(rows)
    option_success = sum(bool(row["option_chain_passed"]) for row in rows)
    parent_success = sum(bool(row["parent_chain_passed"]) for row in rows)
    bridge_start_times = [
        row["option"]["result"]["shooter_causal_strike_bridge_start_time_sec"] for row in rows
    ]
    bridge_target_velocities = [
        row["option"]["result"]["shooter_causal_strike_bridge_peak_target_velocity_rms_rad_s"]
        for row in rows
    ]
    metrics = {
        "case_count": count,
        "option_chain_success_count": option_success,
        "parent_chain_success_count": parent_success,
        "option_chain_success_rate": option_success / count,
        "parent_chain_success_rate": parent_success / count,
        "chain_success_gain": option_success - parent_success,
        "option_safe_rate": sum(bool(row["option_safe"]) for row in rows) / count,
        "parent_safe_rate": sum(bool(row["parent_safe"]) for row in rows) / count,
        "exact_replay_rate": sum(bool(row["exact_replay"]) for row in rows) / count,
        "option_goal_count": sum(bool(row["option"]["result"]["goal_crossed"]) for row in rows),
        "option_save_count": sum(
            bool(row["option"]["result"]["goalkeeper_save_observed"]) for row in rows
        ),
        "mean_option_shot_speed_mps": float(
            np.mean([row["option"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "mean_parent_shot_speed_mps": float(
            np.mean([row["parent"]["result"]["shot_peak_ball_speed_mps"] for row in rows])
        ),
        "minimum_option_bridge_start_time_sec": min(
            float(value) if isinstance(value, int | float) else 0.0 for value in bridge_start_times
        ),
        "maximum_option_bridge_target_velocity_rms_rad_s": max(
            float(value) if isinstance(value, int | float) else float("inf")
            for value in bridge_target_velocities
        ),
        "maximum_option_root_step_m": max(
            float(row["option"]["quality"]["maximum_root_step_m"]) for row in rows
        ),
        "maximum_option_ball_step_m": max(
            float(row["option"]["quality"]["maximum_ball_step_m"]) for row in rows
        ),
    }
    gates = {
        "all_options_committed": all(
            row["option"]["result"]["shooter_causal_strike_bridge_started"] for row in rows
        ),
        "all_options_reached_contact_or_recovery": all(
            row["option"]["result"]["shooter_causal_strike_option_final_phase"] == "RECOVER"
            for row in rows
        ),
        "option_safe_rate": metrics["option_safe_rate"] == 1.0,
        "parent_safe_rate": metrics["parent_safe_rate"] == 1.0,
        "exact_replay_rate": metrics["exact_replay_rate"] == 1.0,
        "option_success_rate": option_success / count + 1.0e-12
        >= quality.minimum_actor_success_rate,
        "option_not_worse_than_parent": option_success >= parent_success,
        "option_has_measured_gain": option_success
        >= parent_success + quality.minimum_success_gain_cases,
        "both_goal_and_save_outcomes": metrics["option_goal_count"] >= 1
        and metrics["option_save_count"] >= 1,
        "delayed_causal_commit": metrics["minimum_option_bridge_start_time_sec"] >= 3.0,
        "bounded_bridge_target_velocity": (
            metrics["maximum_option_bridge_target_velocity_rms_rad_s"] <= 2.0
        ),
        "continuous_root_state": metrics["maximum_option_root_step_m"]
        <= quality.maximum_root_step_m,
        "continuous_ball_state": metrics["maximum_option_ball_step_m"]
        <= quality.maximum_ball_step_m,
    }
    passed = all(gates.values())
    status = (
        "PASS_CAUSAL_STRIKE_OPTION_RETENTION"
        if sealed and passed
        else (
            "PASS_CAUSAL_STRIKE_OPTION_DEVELOPMENT" if passed else "REJECTED_CAUSAL_STRIKE_OPTION"
        )
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.causal_strike_option_exam.v1",
        "status": status,
        "sealed": sealed,
        "partition": "SEALED_HOLDOUT" if sealed else "DEVELOPMENT_REPLAY",
        "promotion_eligible": sealed and passed,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "option_config_hash": option.config_hash,
        "source_s95_policy_hash": source_lead["policy_hash"],
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
            "option_pose_joint_torque_or_ball_authority": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "exam-report.json", report)
    return report


def validate_causal_strike_option_retention(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal strike option retention report must be an object")
    claimed = payload.pop("report_hash", None)
    try:
        gates = payload.get("gates")
        if (
            claimed != hash_json(payload)
            or payload.get("schema_version") != "rosclaw.growth.causal_strike_option_exam.v1"
            or payload.get("status") != "PASS_CAUSAL_STRIKE_OPTION_RETENTION"
            or payload.get("sealed") is not True
            or payload.get("promotion_eligible") is not True
            or not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
            or payload.get("implementation_hash") != _implementation_hash()
        ):
            raise ValueError("causal strike option retention authority is invalid")
        boundary = payload.get("evidence_boundary", {})
        if (
            boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("option_pose_joint_torque_or_ball_authority") is not False
        ):
            raise ValueError("causal strike option evidence boundary is invalid")
        request = source.parent / "request.json"
        if not request.is_file() or hash_bytes(request.read_bytes()) != payload.get("request_hash"):
            raise ValueError("causal strike option request binding changed")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError("causal strike option retention rows are invalid")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("causal strike option retention row is invalid")
            case_dir = source.parent / f"case-{index:03d}"
            for key in ("option_artifact", "replay_artifact", "parent_artifact"):
                artifact = row[key]
                artifact_path = case_dir / artifact["file"]
                if (
                    not artifact_path.is_file()
                    or hash_bytes(artifact_path.read_bytes()) != artifact["file_hash"]
                ):
                    raise ValueError("causal strike option trajectory binding changed")
    finally:
        if claimed is not None:
            payload["report_hash"] = claimed
    return payload


def _run_case(
    job: tuple[
        Path,
        Path,
        Path,
        int,
        CausalTransitionContext,
        G1CausalStrikeOptionConfig,
        CausalTransitionGrowthConfig,
    ],
) -> dict[str, Any]:
    asset_root, source_s95_dir, output, index, context, option, quality = job
    lead_policy, _ = _load_lead_policy(source_s95_dir)
    option_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    option_kwargs["shooter_causal_strike_option_config"] = option
    parent_kwargs = _context_kwargs(
        lead_policy=lead_policy,
        config=quality,
        context=context,
        receiver_start_sec=quality.parent_receiver_start_sec,
    )
    option_result, option_trajectory = simulate_shared_world(asset_root, **option_kwargs)
    replay_result, replay_trajectory = simulate_shared_world(asset_root, **option_kwargs)
    parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
    case_dir = output / f"case-{index:03d}"
    case_dir.mkdir(parents=True)
    option_artifact = _save_trajectory(case_dir / "option-primary.npz", option_trajectory)
    replay_artifact = _save_trajectory(case_dir / "option-replay.npz", replay_trajectory)
    parent_artifact = _save_trajectory(case_dir / "parent.npz", parent_trajectory)
    option_quality = _chain_quality(option_result, option_trajectory, quality)
    parent_quality = _chain_quality(parent_result, parent_trajectory, quality)
    return {
        "case_id": context.case_id,
        "context": asdict(context),
        "context_hash": context.context_hash,
        "option": {"result": option_result.to_dict(), "quality": option_quality},
        "parent": {"result": parent_result.to_dict(), "quality": parent_quality},
        "option_chain_passed": option_quality["chain_passed"],
        "parent_chain_passed": parent_quality["chain_passed"],
        "option_safe": option_quality["safe"],
        "parent_safe": parent_quality["safe"],
        "exact_replay": bool(
            option_result.to_dict() == replay_result.to_dict()
            and option_artifact["trajectory_digest"] == replay_artifact["trajectory_digest"]
        ),
        "option_artifact": option_artifact,
        "replay_artifact": replay_artifact,
        "parent_artifact": parent_artifact,
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "causal_strike_option.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parents[1] / "providers" / "g1" / "transition_bridge.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("causal strike option output must be new and external")
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
    "default_causal_strike_option_development_contexts",
    "default_causal_strike_option_holdouts",
    "run_causal_strike_option_exam",
    "validate_causal_strike_option_retention",
]
