"""S205 repair one failed finish basin with content-bound active search.

This stage is deliberately narrower than a deployable contextual actor.  It
turns a rejected S204 holdout into a durable failure-feedback example, searches
four bounded high-level intent parameters in CPU MuJoCo, and requires an exact
independent replay.  It never emits joint positions, torques, ROS, DDS, vendor,
or hardware commands and cannot authorize promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.bounded_active_search import (
    BoundedActiveSearchPlan,
    BoundedSearchCandidate,
    BoundedSearchDimension,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult, simulate_shared_world
from rosclaw_soccer.training.contextual_finish_portfolio import (
    ContextualFinishPortfolioConfig,
    _context_kwargs,
    _control_config_from_dict,
    _run_jobs,
    _safe_result_dict,
    _stability_retained,
    validate_contextual_finish_portfolio,
)
from rosclaw_soccer.training.contextual_finish_portfolio import (
    _config_from_dict as _portfolio_config_from_dict,
)
from rosclaw_soccer.training.contextual_finish_target_growth import (
    ContextualFinishTargetGrowthConfig,
    _safe,
    _save_trajectory,
    _validate_trajectory,
)


@dataclass(frozen=True)
class ExtendedFinishIntentRepairConfig:
    """A fixed, reviewable trust region around one diagnostic warm start."""

    source_case_id: str = "basin-a-sealed"
    policy_target_y_bounds_m: tuple[float, float] = (0.23, 0.31)
    foot_yaw_bounds_rad: tuple[float, float] = (0.01, 0.09)
    stance_offset_y_bounds_m: tuple[float, float] = (-0.08, 0.02)
    foot_pitch_bounds_rad: tuple[float, float] = (-0.02, 0.08)
    warm_start: tuple[float, float, float, float] = (
        0.29327380962483585,
        0.05210846198908985,
        -0.03931454807892441,
        0.035814595706760884,
    )
    local_candidate_count: int = 32
    sequence_skip: int = 96
    local_radius_fraction: float = 0.25
    refinement_sequence_skip: int = 192
    refinement_radius_fraction: float = 0.125
    maximum_target_error_m: float = 0.10
    maximum_pass_error_m: float = 0.05
    minimum_error_improvement_m: float = 0.20
    simulation_duration_sec: float = 10.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.extended_finish_intent_repair_config.v1"

    def __post_init__(self) -> None:
        bounds = (
            self.policy_target_y_bounds_m,
            self.foot_yaw_bounds_rad,
            self.stance_offset_y_bounds_m,
            self.foot_pitch_bounds_rad,
        )
        if (
            self.source_case_id != "basin-a-sealed"
            or any(
                len(bound) != 2
                or not all(math.isfinite(value) for value in bound)
                or bound[0] >= bound[1]
                for bound in bounds
            )
            or len(self.warm_start) != 4
            or not all(math.isfinite(value) for value in self.warm_start)
            or any(
                not bound[0] <= value <= bound[1]
                for value, bound in zip(self.warm_start, bounds, strict=True)
            )
            or self.local_candidate_count != 32
            or self.sequence_skip != 96
            or self.local_radius_fraction != 0.25
            or self.refinement_sequence_skip != 192
            or self.refinement_radius_fraction != 0.125
            or not 0.05 <= self.maximum_target_error_m <= 0.10
            or not 0.01 <= self.maximum_pass_error_m <= 0.05
            or not 0.10 <= self.minimum_error_improvement_m <= 0.50
            or not 8.0 <= self.simulation_duration_sec <= 15.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.extended_finish_intent_repair_config.v1"
        ):
            raise ValueError("extended finish intent repair config is invalid")

    @property
    def plan(self) -> BoundedActiveSearchPlan:
        return self._plan(
            sequence_skip=self.sequence_skip,
            radius_fraction=self.local_radius_fraction,
        )

    @property
    def refinement_plan(self) -> BoundedActiveSearchPlan:
        return self._plan(
            sequence_skip=self.refinement_sequence_skip,
            radius_fraction=self.refinement_radius_fraction,
        )

    def _plan(self, *, sequence_skip: int, radius_fraction: float) -> BoundedActiveSearchPlan:
        names = (
            "policy_target_y_m",
            "foot_yaw_offset_rad",
            "stance_offset_y_m",
            "foot_pitch_offset_rad",
        )
        bounds = (
            self.policy_target_y_bounds_m,
            self.foot_yaw_bounds_rad,
            self.stance_offset_y_bounds_m,
            self.foot_pitch_bounds_rad,
        )
        return BoundedActiveSearchPlan(
            dimensions=tuple(
                BoundedSearchDimension(name, bound[0], bound[1])
                for name, bound in zip(names, bounds, strict=True)
            ),
            global_candidate_count=64,
            local_candidate_count=self.local_candidate_count,
            sequence_skip=sequence_skip,
            local_radius_fraction=radius_fraction,
        )

    @property
    def config_hash(self) -> str:
        return cast(str, hash_json(asdict(self)))


def run_extended_finish_intent_repair(
    *,
    asset_root: Path,
    source_s204_path: Path,
    source_checkout: Path,
    output_dir: Path,
    config: ExtendedFinishIntentRepairConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Search a bounded four-dimensional repair for an S204 failed holdout."""

    if not 1 <= workers <= 4:
        raise ValueError("S205 workers must be in [1, 4]")
    active = config or ExtendedFinishIntentRepairConfig()
    source_path = source_s204_path.expanduser().resolve()
    source = validate_contextual_finish_portfolio(source_path)
    source_request_path = source_path.parent / "request.json"
    source_request = json.loads(source_request_path.read_text(encoding="utf-8"))
    holdout = cast(dict[str, Any], source["holdouts"][active.source_case_id])
    source_result = cast(dict[str, Any], holdout["result"])
    source_error = source_result.get("target_error_m")
    if (
        source.get("status") != "REJECTED_CONTEXTUAL_FINISH_PORTFOLIO"
        or source.get("passed") is not False
        or holdout.get("expected") != "PRECISE_EXECUTION"
        or holdout.get("candidate_action_executed") is not True
        or holdout.get("passed") is not False
        or not isinstance(source_error, int | float)
        or isinstance(source_error, bool)
        or not math.isfinite(float(source_error))
        or float(source_error) <= active.maximum_target_error_m
    ):
        raise ValueError("S205 requires a rejected finite S204 precision holdout")
    output = _new_output(output_dir, source_checkout)
    context = cast(
        dict[str, Any],
        source_request["resolved_contexts"][active.source_case_id],
    )
    controller = _control_config_from_dict(cast(dict[str, Any], source_request["control_config"]))
    portfolio_config = _portfolio_config_from_dict(cast(dict[str, Any], source_request["config"]))
    physical_target = cast(tuple[float, float, float], tuple(source_request["physical_target_m"]))
    plan = active.plan
    planned = plan.local_candidates(active.warm_start)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.extended_finish_intent_repair_request.v1",
        "source_commit": _git_head(source_checkout),
        "source_s204_hash": source["report_hash"],
        "source_s204_file_hash": hash_bytes(source_path.read_bytes()),
        "source_s204_request_hash": hash_bytes(source_request_path.read_bytes()),
        "source_case_id": active.source_case_id,
        "source_context_hash": holdout["context_hash"],
        "source_failed_trajectory_hash": holdout["trajectory"]["trajectory_digest"],
        "source_failed_target_error_m": source_error,
        "physical_target_m": list(physical_target),
        "context": context,
        "control_config": source_request["control_config"],
        "control_envelope_hash": source_request["control_envelope_hash"],
        "config": asdict(active),
        "config_hash": active.config_hash,
        "search_plan": asdict(plan),
        "search_plan_hash": plan.plan_hash,
        "coarse_candidate_hashes": [candidate.candidate_hash for candidate in planned],
        "refinement_plan": asdict(active.refinement_plan),
        "refinement_plan_hash": active.refinement_plan.plan_hash,
        "refinement_seed_rule": "BEST_SAFE_STABLE_GOAL_THEN_ERROR",
        "implementation_hash": _implementation_hash(),
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "direct_joint_torque_output": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)

    baseline_target = cast(tuple[float, float, float], tuple(holdout["executed_policy_target_m"]))
    baseline_yaw = float(holdout["executed_foot_yaw_offset_rad"])
    baseline_kwargs = _context_kwargs(
        context_record=context,
        target=baseline_target,
        foot_yaw=baseline_yaw,
        controller=controller,
        duration=active.simulation_duration_sec,
    )
    baseline_result, baseline_trajectory = simulate_shared_world(asset_root, **baseline_kwargs)
    baseline_record = _save_trajectory(output / "source-failure-replay.npz", baseline_trajectory)
    baseline_exact = bool(
        baseline_result.to_dict() == source_result
        and baseline_record["trajectory_digest"] == holdout["trajectory"]["trajectory_digest"]
    )

    jobs = [
        (
            asset_root.expanduser().resolve(),
            _candidate_kwargs(
                context=context,
                controller=controller,
                duration=active.simulation_duration_sec,
                physical_target=physical_target,
                values=candidate.values,
            ),
        )
        for candidate in planned
    ]
    outcomes = _run_jobs(jobs, workers)
    coarse_rows = [
        _candidate_row(
            output=output,
            search_stage="COARSE",
            candidate=candidate,
            result=result,
            trajectory=trajectory,
            config=active,
            parent_result=cast(dict[str, Any], holdout["parent"]["result"]),
            portfolio_config=portfolio_config,
        )
        for candidate, (result, trajectory) in zip(planned, outcomes, strict=True)
    ]
    refinement_seed = min(coarse_rows, key=_refinement_seed_key)
    refinement_center = cast(tuple[float, ...], tuple(refinement_seed["action_values"]))
    refinement_planned = active.refinement_plan.local_candidates(refinement_center)
    refinement_jobs = [
        (
            asset_root.expanduser().resolve(),
            _candidate_kwargs(
                context=context,
                controller=controller,
                duration=active.simulation_duration_sec,
                physical_target=physical_target,
                values=candidate.values,
            ),
        )
        for candidate in refinement_planned
    ]
    refinement_outcomes = _run_jobs(refinement_jobs, workers)
    refinement_rows = [
        _candidate_row(
            output=output,
            search_stage="REFINEMENT",
            candidate=candidate,
            result=result,
            trajectory=trajectory,
            config=active,
            parent_result=cast(dict[str, Any], holdout["parent"]["result"]),
            portfolio_config=portfolio_config,
        )
        for candidate, (result, trajectory) in zip(
            refinement_planned, refinement_outcomes, strict=True
        )
    ]
    rows = [*coarse_rows, *refinement_rows]
    all_planned = (*planned, *refinement_planned)
    selected = min(rows, key=_selection_key)
    selected_values = cast(tuple[float, ...], tuple(selected["action_values"]))
    replay_result, replay_trajectory = simulate_shared_world(
        asset_root,
        **_candidate_kwargs(
            context=context,
            controller=controller,
            duration=active.simulation_duration_sec,
            physical_target=physical_target,
            values=selected_values,
        ),
    )
    replay_record = _save_trajectory(output / "selected-independent-replay.npz", replay_trajectory)
    exact_replay = bool(
        selected["result"] == replay_result.to_dict()
        and selected["trajectory"]["trajectory_digest"] == replay_record["trajectory_digest"]
    )
    selected_result = cast(dict[str, Any], selected["result"])
    source_parent_result = cast(dict[str, Any], holdout["parent"]["result"])
    stability = _stability_retained(selected_result, source_parent_result, portfolio_config)
    improvement = float(source_error) - float(selected_result["target_error_m"])
    memory_body: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.extended_finish_intent_memory.v1",
        "source_s204_hash": source["report_hash"],
        "source_failure_context_hash": holdout["context_hash"],
        "source_failure_trajectory_hash": holdout["trajectory"]["trajectory_digest"],
        "search_plan_hashes": [plan.plan_hash, active.refinement_plan.plan_hash],
        "selected_candidate_hash": selected["candidate_hash"],
        "selected_trajectory_hash": selected["trajectory"]["trajectory_digest"],
        "action_names": [dimension.name for dimension in plan.dimensions],
        "action_values": list(selected_values),
        "observed_target_error_m": selected_result["target_error_m"],
        "error_improvement_m": improvement,
        "safe": selected["safe"],
        "precise": selected["precise"],
        "stability_retained": selected["stability_retained"],
        "exact_replay": exact_replay,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "direct_joint_torque_output": False,
        "promotion_authorized": False,
    }
    memory_body["memory_hash"] = hash_json(memory_body)
    memory_path = output / "extended-finish-intent-memory.json"
    _write_json(memory_path, memory_body)
    gates = {
        "source_rejection_bound": True,
        "source_failure_exactly_replayed": baseline_exact,
        "bounded_plan_content_bound": all(
            row["candidate_hash"] == candidate.candidate_hash
            for row, candidate in zip(rows, all_planned, strict=True)
        ),
        "all_candidates_physically_scored": len(rows) == len(all_planned)
        and all(
            isinstance(row["trajectory"].get("trajectory_digest"), str)
            and isinstance(row.get("safe"), bool)
            for row in rows
        ),
        "selected_precise_safe": bool(selected["precise"]),
        "selected_pass_delivery_retained": bool(
            isinstance(selected_result.get("pass_delivery_error_m"), int | float)
            and not isinstance(selected_result.get("pass_delivery_error_m"), bool)
            and float(selected_result["pass_delivery_error_m"]) <= active.maximum_pass_error_m
        ),
        "selected_stability_retained": stability,
        "independent_exact_replay": exact_replay,
        "minimum_error_improvement_met": improvement >= active.minimum_error_improvement_m,
        "memory_content_bound": memory_body["memory_hash"]
        == hash_json({key: value for key, value in memory_body.items() if key != "memory_hash"}),
        "sim_only_no_torque_authority": True,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.extended_finish_intent_repair.v1",
        "status": (
            "PASS_EXTENDED_FINISH_INTENT_REPAIR"
            if passed
            else "REJECTED_EXTENDED_FINISH_INTENT_REPAIR"
        ),
        "passed": passed,
        "promotion_eligible": False,
        "source_s204_hash": source["report_hash"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "baseline": {
            "result": baseline_result.to_dict(),
            "trajectory": baseline_record,
            "matches_source_failure": baseline_exact,
        },
        "refinement": {
            "seed_candidate_hash": refinement_seed["candidate_hash"],
            "center": list(refinement_center),
            "plan_hash": active.refinement_plan.plan_hash,
            "candidate_hashes": [candidate.candidate_hash for candidate in refinement_planned],
        },
        "candidates": rows,
        "selected": selected,
        "selected_replay": {
            "result": replay_result.to_dict(),
            "trajectory": replay_record,
        },
        "memory": {
            "file": memory_path.name,
            "file_hash": hash_bytes(memory_path.read_bytes()),
            "memory_hash": memory_body["memory_hash"],
        },
        "metrics": {
            "candidate_count": len(rows),
            "safe_candidate_count": sum(bool(row["safe"]) for row in rows),
            "precise_safe_candidate_count": sum(bool(row["precise"]) for row in rows),
            "eligible_candidate_count": sum(bool(row["eligible"]) for row in rows),
            "source_failed_target_error_m": source_error,
            "selected_target_error_m": selected_result["target_error_m"],
            "error_improvement_m": improvement,
            "selected_support_foot_slip_m": selected_result[
                "shooter_post_contact_support_foot_slip_m"
            ],
            "selected_min_pelvis_height_m": selected_result["shooter_min_pelvis_height_m"],
        },
        "gates": gates,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "scope": "ONE_FAILED_CONTEXT_POINT_REPAIR_NOT_GENERALIZATION",
            "physics_authority": "CPU_MUJOCO",
            "selection_uses_rendered_pixels": False,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "direct_joint_torque_output": False,
            "promotion_authorized": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "extended-finish-intent-repair.json", report)
    return report


def validate_extended_finish_intent_repair(path: Path) -> dict[str, Any]:
    """Reconstruct S205 selection, memory and all content commitments."""

    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S205 report must be an object")
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = report_path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        active = _config_from_dict(cast(dict[str, Any], request["config"]))
        if (
            expected_hash != hash_json(payload)
            or payload.get("implementation_hash") != _implementation_hash()
            or request.get("implementation_hash") != _implementation_hash()
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or request.get("config_hash") != active.config_hash
            or request.get("search_plan_hash") != active.plan.plan_hash
            or hash_json(request.get("search_plan")) != active.plan.plan_hash
            or request.get("refinement_plan_hash") != active.refinement_plan.plan_hash
            or hash_json(request.get("refinement_plan")) != active.refinement_plan.plan_hash
            or request.get("refinement_seed_rule") != "BEST_SAFE_STABLE_GOAL_THEN_ERROR"
            or request.get("physics_authority") != "CPU_MUJOCO"
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or request.get("direct_joint_torque_output") is not False
            or request.get("pixels_used_for_scoring") is not False
        ):
            raise ValueError("S205 report integrity changed")
        source_path = _source_s204_path(request, report_path)
        source = validate_contextual_finish_portfolio(source_path)
        source_request_path = source_path.parent / "request.json"
        source_request = json.loads(source_request_path.read_text(encoding="utf-8"))
        holdout = cast(dict[str, Any], source["holdouts"][active.source_case_id])
        if (
            source.get("report_hash") != request.get("source_s204_hash")
            or payload.get("source_s204_hash") != source.get("report_hash")
            or hash_bytes(source_path.read_bytes()) != request.get("source_s204_file_hash")
            or hash_bytes(source_request_path.read_bytes())
            != request.get("source_s204_request_hash")
            or source.get("status") != "REJECTED_CONTEXTUAL_FINISH_PORTFOLIO"
            or holdout.get("passed") is not False
            or request.get("source_context_hash") != holdout.get("context_hash")
            or request.get("source_failed_trajectory_hash")
            != holdout["trajectory"]["trajectory_digest"]
            or request.get("source_failed_target_error_m") != holdout["result"]["target_error_m"]
            or request.get("context") != source_request["resolved_contexts"][active.source_case_id]
            or request.get("control_config") != source_request.get("control_config")
            or request.get("control_envelope_hash") != source_request.get("control_envelope_hash")
            or request.get("physical_target_m") != source_request.get("physical_target_m")
        ):
            raise ValueError("S205 source failure binding changed")
        baseline = cast(dict[str, Any], payload["baseline"])
        _validate_trajectory(report_path.parent, baseline["trajectory"])
        baseline_exact = bool(
            baseline["result"] == holdout["result"]
            and baseline["trajectory"]["trajectory_digest"]
            == holdout["trajectory"]["trajectory_digest"]
        )
        if baseline.get("matches_source_failure") is not baseline_exact:
            raise ValueError("S205 source replay derivation changed")
        planned = active.plan.local_candidates(active.warm_start)
        if request.get("coarse_candidate_hashes") != [
            candidate.candidate_hash for candidate in planned
        ]:
            raise ValueError("S205 planned candidate commitment changed")
        rows = cast(list[dict[str, Any]], payload["candidates"])
        if len(rows) != 2 * len(planned):
            raise ValueError("S205 candidate count changed")
        coarse_rows = rows[: len(planned)]
        for row, candidate in zip(coarse_rows, planned, strict=True):
            _validate_candidate_row(
                report_path.parent,
                row,
                candidate,
                "COARSE",
                active,
                cast(dict[str, Any], holdout["parent"]["result"]),
                _portfolio_config_from_dict(cast(dict[str, Any], source_request["config"])),
            )
        refinement_seed = min(coarse_rows, key=_refinement_seed_key)
        refinement_center = cast(tuple[float, ...], tuple(refinement_seed["action_values"]))
        refinement_planned = active.refinement_plan.local_candidates(refinement_center)
        refinement = cast(dict[str, Any], payload["refinement"])
        if refinement != {
            "seed_candidate_hash": refinement_seed["candidate_hash"],
            "center": list(refinement_center),
            "plan_hash": active.refinement_plan.plan_hash,
            "candidate_hashes": [candidate.candidate_hash for candidate in refinement_planned],
        }:
            raise ValueError("S205 refinement derivation changed")
        refinement_rows = rows[len(planned) :]
        for row, candidate in zip(refinement_rows, refinement_planned, strict=True):
            _validate_candidate_row(
                report_path.parent,
                row,
                candidate,
                "REFINEMENT",
                active,
                cast(dict[str, Any], holdout["parent"]["result"]),
                _portfolio_config_from_dict(cast(dict[str, Any], source_request["config"])),
            )
        all_planned = (*planned, *refinement_planned)
        selected = min(rows, key=_selection_key)
        if payload.get("selected") != selected:
            raise ValueError("S205 selected candidate changed")
        replay = cast(dict[str, Any], payload["selected_replay"])
        _validate_trajectory(report_path.parent, replay["trajectory"])
        exact_replay = bool(
            selected["result"] == replay["result"]
            and selected["trajectory"]["trajectory_digest"]
            == replay["trajectory"]["trajectory_digest"]
        )
        portfolio_config = _portfolio_config_from_dict(
            cast(dict[str, Any], source_request["config"])
        )
        selected_result = cast(dict[str, Any], selected["result"])
        stability = _stability_retained(
            selected_result,
            cast(dict[str, Any], holdout["parent"]["result"]),
            portfolio_config,
        )
        source_error = float(holdout["result"]["target_error_m"])
        improvement = source_error - float(selected_result["target_error_m"])
        memory_record = cast(dict[str, Any], payload["memory"])
        memory_path = report_path.parent / str(memory_record["file"])
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        memory_hash = memory.pop("memory_hash", None)
        expected_memory = {
            "schema_version": "rosclaw_soccer.extended_finish_intent_memory.v1",
            "source_s204_hash": source["report_hash"],
            "source_failure_context_hash": holdout["context_hash"],
            "source_failure_trajectory_hash": holdout["trajectory"]["trajectory_digest"],
            "search_plan_hashes": [
                active.plan.plan_hash,
                active.refinement_plan.plan_hash,
            ],
            "selected_candidate_hash": selected["candidate_hash"],
            "selected_trajectory_hash": selected["trajectory"]["trajectory_digest"],
            "action_names": [dimension.name for dimension in active.plan.dimensions],
            "action_values": selected["action_values"],
            "observed_target_error_m": selected_result["target_error_m"],
            "error_improvement_m": improvement,
            "safe": selected["safe"],
            "precise": selected["precise"],
            "stability_retained": selected["stability_retained"],
            "exact_replay": exact_replay,
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "direct_joint_torque_output": False,
            "promotion_authorized": False,
        }
        if (
            memory != expected_memory
            or memory_hash != hash_json(memory)
            or memory_record.get("memory_hash") != memory_hash
            or memory_record.get("file_hash") != hash_bytes(memory_path.read_bytes())
        ):
            raise ValueError("S205 repair memory binding changed")
        memory["memory_hash"] = memory_hash
        gates = {
            "source_rejection_bound": True,
            "source_failure_exactly_replayed": baseline_exact,
            "bounded_plan_content_bound": all(
                row["candidate_hash"] == candidate.candidate_hash
                for row, candidate in zip(rows, all_planned, strict=True)
            ),
            "all_candidates_physically_scored": len(rows) == len(all_planned)
            and all(
                isinstance(row["trajectory"].get("trajectory_digest"), str)
                and isinstance(row.get("safe"), bool)
                for row in rows
            ),
            "selected_precise_safe": bool(selected["precise"]),
            "selected_pass_delivery_retained": bool(
                isinstance(selected_result.get("pass_delivery_error_m"), int | float)
                and not isinstance(selected_result.get("pass_delivery_error_m"), bool)
                and float(selected_result["pass_delivery_error_m"]) <= active.maximum_pass_error_m
            ),
            "selected_stability_retained": stability,
            "independent_exact_replay": exact_replay,
            "minimum_error_improvement_met": improvement >= active.minimum_error_improvement_m,
            "memory_content_bound": memory_hash == hash_json(expected_memory),
            "sim_only_no_torque_authority": True,
        }
        passed = all(gates.values())
        metrics = {
            "candidate_count": len(rows),
            "safe_candidate_count": sum(bool(row["safe"]) for row in rows),
            "precise_safe_candidate_count": sum(bool(row["precise"]) for row in rows),
            "eligible_candidate_count": sum(bool(row["eligible"]) for row in rows),
            "source_failed_target_error_m": source_error,
            "selected_target_error_m": selected_result["target_error_m"],
            "error_improvement_m": improvement,
            "selected_support_foot_slip_m": selected_result[
                "shooter_post_contact_support_foot_slip_m"
            ],
            "selected_min_pelvis_height_m": selected_result["shooter_min_pelvis_height_m"],
        }
        boundary = cast(dict[str, Any], payload.get("evidence_boundary", {}))
        if (
            payload.get("gates") != gates
            or payload.get("metrics") != metrics
            or payload.get("passed") is not passed
            or payload.get("status")
            != (
                "PASS_EXTENDED_FINISH_INTENT_REPAIR"
                if passed
                else "REJECTED_EXTENDED_FINISH_INTENT_REPAIR"
            )
            or payload.get("promotion_eligible") is not False
            or boundary.get("scope") != "ONE_FAILED_CONTEXT_POINT_REPAIR_NOT_GENERALIZATION"
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("selection_uses_rendered_pixels") is not False
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("direct_joint_torque_output") is not False
            or boundary.get("promotion_authorized") is not False
        ):
            raise ValueError("S205 derived gates or authority changed")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return payload


def _candidate_kwargs(
    *,
    context: dict[str, Any],
    controller: ContextualFinishTargetGrowthConfig,
    duration: float,
    physical_target: tuple[float, float, float],
    values: tuple[float, ...],
) -> dict[str, Any]:
    if len(values) != 4:
        raise ValueError("S205 candidate must contain four high-level values")
    target_y, foot_yaw, stance_y, foot_pitch = values
    kwargs = cast(
        dict[str, Any],
        _context_kwargs(
            context_record=context,
            target=(physical_target[0], target_y, float(controller.policy_target_z_m)),
            foot_yaw=foot_yaw,
            controller=controller,
            duration=duration,
        ),
    )
    parameters = dict(kwargs["shooter_parameter_overrides"])
    parameters.update(stance_offset_y=stance_y, foot_pitch_offset=foot_pitch)
    kwargs["shooter_parameter_overrides"] = parameters
    return kwargs


def _candidate_row(
    *,
    output: Path,
    search_stage: str,
    candidate: BoundedSearchCandidate,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    config: ExtendedFinishIntentRepairConfig,
    parent_result: dict[str, Any],
    portfolio_config: ContextualFinishPortfolioConfig,
) -> dict[str, Any]:
    if search_stage not in {"COARSE", "REFINEMENT"}:
        raise ValueError("S205 search stage is invalid")
    result_dict = result.to_dict()
    stability = _stability_retained(result_dict, parent_result, portfolio_config)
    pass_retained = bool(
        isinstance(result_dict.get("pass_delivery_error_m"), int | float)
        and not isinstance(result_dict.get("pass_delivery_error_m"), bool)
        and float(result_dict["pass_delivery_error_m"]) <= config.maximum_pass_error_m
    )
    precise = _precise_result(result_dict, config)
    return {
        "search_stage": search_stage,
        "candidate": asdict(candidate),
        "candidate_hash": candidate.candidate_hash,
        "action_values": list(candidate.values),
        "result": result_dict,
        "safe": _safe(result),
        "precise": precise,
        "pass_delivery_retained": pass_retained,
        "stability_retained": stability,
        "eligible": precise and pass_retained and stability,
        "trajectory": _save_trajectory(
            output / f"{search_stage.lower()}-candidate-{candidate.candidate_index:03d}.npz",
            trajectory,
        ),
    }


def _validate_candidate_row(
    root: Path,
    row: dict[str, Any],
    candidate: BoundedSearchCandidate,
    search_stage: str,
    config: ExtendedFinishIntentRepairConfig,
    parent_result: dict[str, Any],
    portfolio_config: ContextualFinishPortfolioConfig,
) -> None:
    _validate_trajectory(root, row["trajectory"])
    result = cast(dict[str, Any], row["result"])
    if (
        row.get("search_stage") != search_stage
        or row.get("candidate_hash") != candidate.candidate_hash
        or hash_json(row.get("candidate")) != candidate.candidate_hash
        or row.get("action_values") != list(candidate.values)
        or row.get("safe") is not _safe_result_dict(result)
        or row.get("precise") is not _precise_result(result, config)
        or row.get("pass_delivery_retained")
        is not bool(
            isinstance(result.get("pass_delivery_error_m"), int | float)
            and not isinstance(result.get("pass_delivery_error_m"), bool)
            and float(result["pass_delivery_error_m"]) <= config.maximum_pass_error_m
        )
        or row.get("stability_retained")
        is not _stability_retained(result, parent_result, portfolio_config)
        or row.get("eligible")
        is not bool(
            row.get("precise")
            and row.get("pass_delivery_retained")
            and row.get("stability_retained")
        )
    ):
        raise ValueError("S205 candidate derivation changed")


def _precise_result(result: dict[str, Any], config: ExtendedFinishIntentRepairConfig) -> bool:
    error = result.get("target_error_m")
    return bool(
        _safe_result_dict(result)
        and result.get("goal_crossed") is True
        and isinstance(error, int | float)
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        and float(error) <= config.maximum_target_error_m
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    result = cast(dict[str, Any], row["result"])
    error = result.get("target_error_m")
    pass_error = result.get("pass_delivery_error_m")
    return (
        0.0 if row.get("eligible") is True else 1.0,
        0.0 if row["safe"] else 1.0,
        0.0 if result.get("goal_crossed") is True else 1.0,
        (
            float(error)
            if isinstance(error, int | float) and not isinstance(error, bool)
            else math.inf
        ),
        (
            float(pass_error)
            if isinstance(pass_error, int | float) and not isinstance(pass_error, bool)
            else math.inf
        ),
        float(result.get("shooter_post_contact_support_foot_slip_m", math.inf)),
        -float(result.get("shooter_min_pelvis_height_m", -math.inf)),
        float(row["candidate"]["candidate_index"]),
    )


def _refinement_seed_key(row: dict[str, Any]) -> tuple[float, ...]:
    result = cast(dict[str, Any], row["result"])
    error = result.get("target_error_m")
    return (
        0.0
        if row.get("safe") is True
        and row.get("stability_retained") is True
        and result.get("goal_crossed") is True
        else 1.0,
        0.0 if row.get("safe") is True else 1.0,
        0.0 if row.get("stability_retained") is True else 1.0,
        0.0 if result.get("goal_crossed") is True else 1.0,
        (
            float(error)
            if isinstance(error, int | float) and not isinstance(error, bool)
            else math.inf
        ),
        float(row["candidate"]["candidate_index"]),
    )


def _config_from_dict(value: dict[str, Any]) -> ExtendedFinishIntentRepairConfig:
    return ExtendedFinishIntentRepairConfig(
        **{
            **value,
            "policy_target_y_bounds_m": tuple(value["policy_target_y_bounds_m"]),
            "foot_yaw_bounds_rad": tuple(value["foot_yaw_bounds_rad"]),
            "stance_offset_y_bounds_m": tuple(value["stance_offset_y_bounds_m"]),
            "foot_pitch_bounds_rad": tuple(value["foot_pitch_bounds_rad"]),
            "warm_start": tuple(value["warm_start"]),
        }
    )


def _source_s204_path(request: dict[str, Any], report_path: Path) -> Path:
    source_hash = request.get("source_s204_file_hash")
    candidates = tuple(
        report_path.parents[1].glob(
            "s204-contextual-finish-portfolio-*/contextual-finish-portfolio.json"
        )
    )
    for candidate in candidates:
        if hash_bytes(candidate.read_bytes()) == source_hash:
            return candidate
    raise ValueError("S205 source S204 file is unavailable")


def _runtime_manifest() -> dict[str, str]:
    import mujoco

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "bounded_active_search.py",
        Path(__file__).parent / "contextual_finish_portfolio.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_output(path: Path, checkout: Path) -> Path:
    output = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("S205 evidence output must be new and external")
    output.mkdir(parents=True)
    return output


def _git_head(checkout: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-s204", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_extended_finish_intent_repair(
        asset_root=args.asset_root,
        source_s204_path=args.source_s204,
        source_checkout=args.source_checkout,
        output_dir=args.output,
        workers=args.workers,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExtendedFinishIntentRepairConfig",
    "run_extended_finish_intent_repair",
    "validate_extended_finish_intent_repair",
]
