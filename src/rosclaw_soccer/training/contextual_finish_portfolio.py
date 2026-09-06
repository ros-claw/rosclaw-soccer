"""S204 multi-context finisher portfolio with failure-basin vetoes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.contextual_finish_target import (
    FinishTargetCalibrationSample,
    FinishTargetFailureMemory,
    fit_contextual_finish_target_actor,
    load_contextual_finish_target_actor,
    save_contextual_finish_target_actor,
)
from rosclaw_soccer.growth.dynamic_lead_pass import DynamicLeadPassPolicy
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
    RoleOptionBackendRoute,
    select_role_option_backend,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import prepared_finish_plan_features
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.development_evidence import three_role_development_kwargs
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult, simulate_shared_world
from rosclaw_soccer.training.contextual_finish_target_growth import (
    ContextualFinishTargetGrowthConfig,
    _control_envelope_hash,
    _load_lead_pass,
    _safe,
    _save_trajectory,
    _selection_key,
    _simulation_kwargs,
    _validate_trajectory,
    validate_contextual_finish_target_growth,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)

_CASE_ID = re.compile(r"^[a-z][a-z0-9-]{2,47}$")


@dataclass(frozen=True)
class FinishPortfolioContext:
    case_id: str
    receiver_phase_start_sec: float
    receiver_lateral_lane_m: float
    schema_version: str = "rosclaw_soccer.finish_portfolio_context.v1"

    def __post_init__(self) -> None:
        if (
            not _CASE_ID.fullmatch(self.case_id)
            or not math.isfinite(self.receiver_phase_start_sec)
            or not 1.85 <= self.receiver_phase_start_sec <= 1.95
            or not math.isfinite(self.receiver_lateral_lane_m)
            or not 0.05 <= self.receiver_lateral_lane_m <= 0.12
        ):
            raise ValueError("finish portfolio context is invalid")


@dataclass(frozen=True)
class ContextualFinishPortfolioConfig:
    success_discovery: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("basin-a-recovery", 1.900, 0.07875),
        FinishPortfolioContext("basin-a-left", 1.900, 0.079),
        FinishPortfolioContext("basin-a-right", 1.900, 0.081),
        FinishPortfolioContext("basin-b-left", 1.920, 0.099),
        FinishPortfolioContext("basin-b-recovery", 1.920, 0.09921),
        FinishPortfolioContext("basin-b-right", 1.920, 0.101),
    )
    failure_discovery: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("gap-failure", 1.910, 0.09),
        FinishPortfolioContext("right-failure", 1.930, 0.11),
    )
    success_holdouts: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("basin-a-sealed", 1.900, 0.07880),
        FinishPortfolioContext("basin-b-sealed", 1.920, 0.09925),
    )
    failure_holdouts: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("gap-veto-sealed", 1.911, 0.09),
        FinishPortfolioContext("right-veto-sealed", 1.931, 0.11),
    )
    success_policy_target_y_candidates_m: tuple[float, ...] = (0.245, 0.265, 0.275, 0.285)
    success_foot_yaw_candidates_rad: tuple[float, ...] = (0.04, 0.05, 0.06, 0.07)
    failure_policy_target_y_candidates_m: tuple[float, ...] = (
        0.245,
        0.255,
        0.265,
        0.275,
        0.285,
        0.295,
    )
    failure_foot_yaw_candidates_rad: tuple[float, ...] = (0.04, 0.05, 0.06, 0.07, 0.08)
    policy_target_z_m: float = 0.50
    maximum_target_error_m: float = 0.10
    maximum_pass_error_m: float = 0.05
    maximum_pelvis_regression_m: float = 0.03
    maximum_support_slip_regression_m: float = 0.08
    maximum_absolute_support_slip_m: float = 0.16
    failure_exclusion_distance: float = 0.40
    simulation_duration_sec: float = 10.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        partitions = (
            self.success_discovery,
            self.failure_discovery,
            self.success_holdouts,
            self.failure_holdouts,
        )
        all_contexts = tuple(context for partition in partitions for context in partition)
        coordinates = {
            (context.receiver_phase_start_sec, context.receiver_lateral_lane_m)
            for context in all_contexts
        }
        success_actions = len(self.success_policy_target_y_candidates_m) * len(
            self.success_foot_yaw_candidates_rad
        )
        failure_actions = len(self.failure_policy_target_y_candidates_m) * len(
            self.failure_foot_yaw_candidates_rad
        )
        if (
            tuple(len(partition) for partition in partitions) != (6, 2, 2, 2)
            or len({context.case_id for context in all_contexts}) != len(all_contexts)
            or len(coordinates) != len(all_contexts)
            or len({context.receiver_lateral_lane_m for context in self.success_discovery}) < 2
            or max(context.receiver_phase_start_sec for context in self.success_discovery)
            - min(context.receiver_phase_start_sec for context in self.success_discovery)
            < 0.02
            or success_actions != 16
            or failure_actions != 30
            or len(set(self.success_policy_target_y_candidates_m))
            != len(self.success_policy_target_y_candidates_m)
            or len(set(self.success_foot_yaw_candidates_rad))
            != len(self.success_foot_yaw_candidates_rad)
            or len(set(self.failure_policy_target_y_candidates_m))
            != len(self.failure_policy_target_y_candidates_m)
            or len(set(self.failure_foot_yaw_candidates_rad))
            != len(self.failure_foot_yaw_candidates_rad)
            or not all(
                math.isfinite(value) and abs(value) <= 2.0
                for value in (
                    *self.success_policy_target_y_candidates_m,
                    *self.failure_policy_target_y_candidates_m,
                )
            )
            or not all(
                math.isfinite(value) and abs(value) <= 0.12
                for value in (
                    *self.success_foot_yaw_candidates_rad,
                    *self.failure_foot_yaw_candidates_rad,
                )
            )
            or not 0.05 <= self.policy_target_z_m <= 2.50
            or not 0.05 <= self.maximum_target_error_m <= 0.10
            or not 0.01 <= self.maximum_pass_error_m <= 0.05
            or not 0.0 <= self.maximum_pelvis_regression_m <= 0.10
            or not 0.0 <= self.maximum_support_slip_regression_m <= 0.15
            or not 0.08 <= self.maximum_absolute_support_slip_m <= 0.20
            or not 0.10 <= self.failure_exclusion_distance <= 1.0
            or not 8.0 <= self.simulation_duration_sec <= 15.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("contextual finish portfolio config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_contextual_finish_portfolio(
    *,
    asset_root: Path,
    source_s203_path: Path,
    source_lead_pass_dir: Path,
    source_checkout: Path,
    output_dir: Path,
    config: ContextualFinishPortfolioConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Grow two supported contact basins and preserve known failures as vetoes."""

    if not 1 <= workers <= 4:
        raise ValueError("S204 workers must be in [1, 4]")
    active = config or ContextualFinishPortfolioConfig()
    controller = ContextualFinishTargetGrowthConfig()
    source_path = source_s203_path.expanduser().resolve()
    source = validate_contextual_finish_target_growth(source_path)
    if (
        source.get("status") != "PASS_SINGLE_CONTEXT_FINISH_TARGET_SEED"
        or source.get("promotion_eligible") is not False
        or source.get("actor", {}).get("evidence_ready") is not False
    ):
        raise ValueError("S204 requires the passing, non-deployable S203 seed")
    policy, policy_source = _load_lead_pass(source_lead_pass_dir)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    source_request = json.loads((source_path.parent / "request.json").read_text(encoding="utf-8"))
    control_hash = _control_envelope_hash(controller)
    if (
        source_request.get("body_hash") != qualification.body_hash
        or source_request.get("kick_prior_hash") != qualification.kick_prior_hash
        or source_request.get("control_envelope_hash") != control_hash
        or source_request.get("source_lead_pass_policy_hash") != policy.artifact_hash
    ):
        raise ValueError("S204 source physical lineage changed")
    fixture = build_independent_three_vs_three_fixture(asset_root)
    finisher = next(cell for cell in fixture.cells if cell.agent_id == "red.finisher")
    output = _new_output(output_dir, source_checkout)
    physical_target = tuple(float(value) for value in source_request["physical_target_m"])
    parent_target = tuple(float(value) for value in source_request["parent_policy_target_m"])
    parent_yaw = float(source_request["parent_foot_yaw_offset_rad"])
    resolved = {
        context.case_id: _resolve_context(
            context=context,
            policy=policy,
            physical_target=cast(tuple[float, float, float], physical_target),
        )
        for context in _all_contexts(active)
    }
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_portfolio_request.v1",
        "source_commit": _git_head(source_checkout),
        "source_s203_hash": source["report_hash"],
        "source_s203_file_hash": hash_bytes(source_path.read_bytes()),
        "source_lead_pass_evidence_hash": policy_source["evidence_hash"],
        "source_lead_pass_policy_hash": policy.artifact_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "fixture_hash": fixture.fixture_hash,
        "roster_hash": fixture.roster.roster_hash,
        "finisher_cell_hash": finisher.cell_hash,
        "finisher_self_model_hash": finisher.self_model.self_model_hash,
        "physical_target_m": list(physical_target),
        "parent_policy_target_m": list(parent_target),
        "parent_foot_yaw_offset_rad": parent_yaw,
        "control_config": asdict(controller),
        "control_envelope_hash": control_hash,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "resolved_contexts": resolved,
        "implementation_hash": _implementation_hash(),
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)

    successes = _run_search_partition(
        asset_root=asset_root,
        output=output,
        prefix="success",
        contexts=active.success_discovery,
        resolved=resolved,
        y_candidates=active.success_policy_target_y_candidates_m,
        yaw_candidates=active.success_foot_yaw_candidates_rad,
        physical_target=cast(tuple[float, float, float], physical_target),
        parent_target=cast(tuple[float, float, float], parent_target),
        parent_yaw=parent_yaw,
        controller=controller,
        active=active,
        workers=workers,
    )
    failures = _run_search_partition(
        asset_root=asset_root,
        output=output,
        prefix="failure",
        contexts=active.failure_discovery,
        resolved=resolved,
        y_candidates=active.failure_policy_target_y_candidates_m,
        yaw_candidates=active.failure_foot_yaw_candidates_rad,
        physical_target=cast(tuple[float, float, float], physical_target),
        parent_target=cast(tuple[float, float, float], parent_target),
        parent_yaw=parent_yaw,
        controller=controller,
        active=active,
        workers=workers,
    )
    samples = tuple(
        _success_sample(
            context=context,
            context_record=resolved[context.case_id],
            outcome=cast(dict[str, Any], successes[context.case_id]),
            physical_target=cast(tuple[float, float, float], physical_target),
            control_hash=control_hash,
        )
        for context in active.success_discovery
    )
    failure_memories = tuple(
        _failure_memory(
            context=context,
            context_record=resolved[context.case_id],
            outcome=cast(dict[str, Any], failures[context.case_id]),
            control_hash=control_hash,
        )
        for context in active.failure_discovery
    )
    actor = replace(
        fit_contextual_finish_target_actor(
            body_hash=qualification.body_hash,
            kick_prior_hash=qualification.kick_prior_hash,
            roster_hash=fixture.roster.roster_hash,
            finisher_self_model_hash=finisher.self_model.self_model_hash,
            control_envelope_hash=control_hash,
            source_evidence_hashes=(str(source["report_hash"]),),
            samples=samples,
            failure_memories=failure_memories,
        ),
        nearest_sample_count=1,
        failure_exclusion_distance=active.failure_exclusion_distance,
    )
    actor_path = output / "contextual-finish-portfolio-actor.json"
    save_contextual_finish_target_actor(actor, actor_path)
    holdouts = _run_holdouts(
        asset_root=asset_root,
        output=output,
        actor=actor,
        active=active,
        controller=controller,
        resolved=resolved,
        parent_target=cast(tuple[float, float, float], parent_target),
        parent_yaw=parent_yaw,
    )
    discovery_exact = all(
        case["selected_exact_replay"] for case in (*successes.values(), *failures.values())
    )
    holdout_passed = all(case["passed"] for case in holdouts.values())
    parent_retention = all(case["stability_retained"] for case in successes.values()) and all(
        case.get("stability_retained", True) for case in holdouts.values()
    )
    candidate = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        option=PhysicalSoccerOption.SHOOT,
        artifact_hash=actor.actor_hash,
        evidence_hash=str(source["report_hash"]),
        distinct_context_count=actor.distinct_context_count,
        distinct_trajectory_count=actor.distinct_trajectory_count,
        strict_replay=discovery_exact,
        holdout_passed=holdout_passed,
        parent_retention_passed=parent_retention,
    )
    route = select_role_option_backend(
        cell=finisher,
        option=PhysicalSoccerOption.SHOOT,
        candidates=(candidate,),
    )
    gates = {
        "source_single_context_seed_bound": True,
        "six_distinct_success_contexts": actor.distinct_context_count == 6,
        "multiple_physical_success_lanes": len(
            {context.receiver_lateral_lane_m for context in active.success_discovery}
        )
        >= 2,
        "all_success_discovery_precise_safe": all(case["passed"] for case in successes.values()),
        "bounded_failure_searches_proved": all(
            case["failure_proved"] for case in failures.values()
        ),
        "failure_memories_bound": len(actor.failure_memories) == 2,
        "actor_evidence_ready": actor.evidence_ready,
        "sealed_success_holdouts_passed": all(
            holdouts[context.case_id]["passed"] for context in active.success_holdouts
        ),
        "sealed_failure_holdouts_vetoed": all(
            holdouts[context.case_id]["passed"] for context in active.failure_holdouts
        ),
        "strict_replay_complete": discovery_exact
        and all(case["exact_replay"] for case in holdouts.values()),
        "parent_stability_retained": parent_retention,
        "backend_evidence_ready": candidate.evidence_ready,
        "role_qualified_route_accepted": route.accepted
        and route.selected_backend is RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_portfolio.v1",
        "status": (
            "PASS_CONTEXTUAL_FINISH_PORTFOLIO_AND_FAILURE_VETO"
            if passed
            else "REJECTED_CONTEXTUAL_FINISH_PORTFOLIO"
        ),
        "passed": passed,
        "promotion_eligible": False,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_s203_hash": source["report_hash"],
        "success_discovery": successes,
        "failure_discovery": failures,
        "holdouts": holdouts,
        "actor": {
            "file": actor_path.name,
            "file_hash": hash_bytes(actor_path.read_bytes()),
            "actor_hash": actor.actor_hash,
            "evidence_ready": actor.evidence_ready,
            "distinct_context_count": actor.distinct_context_count,
            "distinct_trajectory_count": actor.distinct_trajectory_count,
            "failure_memory_count": len(actor.failure_memories),
        },
        "backend_candidate": candidate.to_dict(),
        "backend_candidate_hash": candidate.candidate_hash,
        "route": route.to_dict(),
        "route_hash": route.route_hash,
        "gates": gates,
        "metrics": _metrics(successes, failures, holdouts),
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "partition": "DEVELOPMENT_WITH_SEALED_HOLDOUT",
            "plastic_agent_id": "red.finisher",
            "frozen_role_count": 5,
            "joint_torque_owner": "FROZEN_WHOLE_BODY_KICK_PRIOR_WITH_HASHED_SAFETY_ENVELOPE",
            "pixels_used_for_scoring": False,
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "commercial_use_allowed": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "contextual-finish-portfolio.json", report)
    return report


def _run_search_partition(
    *,
    asset_root: Path,
    output: Path,
    prefix: str,
    contexts: tuple[FinishPortfolioContext, ...],
    resolved: dict[str, Any],
    y_candidates: tuple[float, ...],
    yaw_candidates: tuple[float, ...],
    physical_target: tuple[float, float, float],
    parent_target: tuple[float, float, float],
    parent_yaw: float,
    controller: ContextualFinishTargetGrowthConfig,
    active: ContextualFinishPortfolioConfig,
    workers: int,
) -> dict[str, Any]:
    jobs: list[tuple[Path, dict[str, Any]]] = []
    job_keys: list[tuple[str, int, float, float]] = []
    for context in contexts:
        context_record = cast(dict[str, Any], resolved[context.case_id])
        for index, (target_y, foot_yaw) in enumerate(
            (y, yaw) for y in y_candidates for yaw in yaw_candidates
        ):
            jobs.append(
                (
                    asset_root.expanduser().resolve(),
                    _context_kwargs(
                        context_record=context_record,
                        target=(physical_target[0], target_y, active.policy_target_z_m),
                        foot_yaw=foot_yaw,
                        controller=controller,
                        duration=active.simulation_duration_sec,
                    ),
                )
            )
            job_keys.append((context.case_id, index, target_y, foot_yaw))
    outcomes = _run_jobs(jobs, workers)
    grouped: dict[str, list[dict[str, Any]]] = {context.case_id: [] for context in contexts}
    for (case_id, index, target_y, foot_yaw), (result, trajectory) in zip(
        job_keys, outcomes, strict=True
    ):
        record = _save_trajectory(
            output / f"{prefix}-{case_id}-candidate-{index:02d}.npz", trajectory
        )
        grouped[case_id].append(
            {
                "candidate_index": index,
                "policy_target_m": [physical_target[0], target_y, active.policy_target_z_m],
                "foot_yaw_offset_rad": foot_yaw,
                "result": result.to_dict(),
                "safe": _safe(result),
                "trajectory": record,
            }
        )
    values: dict[str, Any] = {}
    base = three_role_development_kwargs()
    for context in contexts:
        context_record = cast(dict[str, Any], resolved[context.case_id])
        rows = grouped[context.case_id]
        selected = min(rows, key=_selection_key)
        selected_result, selected_trajectory = simulate_shared_world(
            asset_root,
            **_context_kwargs(
                context_record=context_record,
                target=cast(tuple[float, float, float], tuple(selected["policy_target_m"])),
                foot_yaw=float(selected["foot_yaw_offset_rad"]),
                controller=controller,
                duration=active.simulation_duration_sec,
            ),
        )
        replay_record = _save_trajectory(
            output / f"{prefix}-{context.case_id}-selected-replay.npz", selected_trajectory
        )
        exact = bool(
            selected["result"] == selected_result.to_dict()
            and selected["trajectory"]["trajectory_digest"] == replay_record["trajectory_digest"]
        )
        parent_result, parent_trajectory = simulate_shared_world(
            asset_root,
            **_context_kwargs(
                context_record=context_record,
                target=parent_target,
                foot_yaw=parent_yaw,
                controller=controller,
                duration=active.simulation_duration_sec,
            ),
        )
        parent = {
            "result": parent_result.to_dict(),
            "safe": _safe(parent_result),
            "trajectory": _save_trajectory(
                output / f"{prefix}-{context.case_id}-parent.npz", parent_trajectory
            ),
        }
        precise_rows = tuple(row for row in rows if _precise(row, active))
        stability = _stability_retained(selected_result.to_dict(), parent_result.to_dict(), active)
        selected_passed = bool(
            _precise(selected, active)
            and exact
            and selected_result.pass_delivery_error_m is not None
            and selected_result.pass_delivery_error_m <= active.maximum_pass_error_m
            and stability
        )
        failure_proved = bool(
            not precise_rows
            and selected["safe"]
            and exact
            and len(rows) == len(y_candidates) * len(yaw_candidates)
        )
        values[context.case_id] = {
            "context": asdict(context),
            "context_hash": context_record["context_hash"],
            "candidate_count": len(rows),
            "safe_candidate_count": sum(bool(row["safe"]) for row in rows),
            "precise_safe_candidate_count": len(precise_rows),
            "rows": rows,
            "selected": selected,
            "selected_replay": {
                "result": selected_result.to_dict(),
                "trajectory": replay_record,
            },
            "selected_exact_replay": exact,
            "parent": parent,
            "stability_retained": stability,
            "failure_proved": failure_proved,
            "passed": selected_passed if prefix == "success" else failure_proved,
            "parent_action": {
                "policy_target_m": list(parent_target),
                "foot_yaw_offset_rad": parent_yaw,
                "source": base["shooter_parameter_overrides"],
            },
        }
    return values


def _run_holdouts(
    *,
    asset_root: Path,
    output: Path,
    actor: Any,
    active: ContextualFinishPortfolioConfig,
    controller: ContextualFinishTargetGrowthConfig,
    resolved: dict[str, Any],
    parent_target: tuple[float, float, float],
    parent_yaw: float,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    success_ids = {context.case_id for context in active.success_holdouts}
    for context in (*active.success_holdouts, *active.failure_holdouts):
        context_record = cast(dict[str, Any], resolved[context.case_id])
        features = cast(tuple[float, ...], tuple(context_record["features"]))
        target = cast(tuple[float, float, float], tuple(context_record["physical_target_m"]))
        decision = actor.decide(features, target)
        expected_success = context.case_id in success_ids
        if expected_success and decision.accepted:
            executed_target = cast(tuple[float, float, float], decision.policy_target_m)
            executed_yaw = cast(float, decision.foot_yaw_offset_rad)
            action_executed = True
        else:
            executed_target = parent_target
            executed_yaw = parent_yaw
            action_executed = False
        kwargs = _context_kwargs(
            context_record=context_record,
            target=executed_target,
            foot_yaw=executed_yaw,
            controller=controller,
            duration=active.simulation_duration_sec,
        )
        result, trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        record = _save_trajectory(output / f"holdout-{context.case_id}.npz", trajectory)
        replay_record = _save_trajectory(
            output / f"holdout-{context.case_id}-replay.npz", replay_trajectory
        )
        exact = bool(
            result.to_dict() == replay_result.to_dict()
            and record["trajectory_digest"] == replay_record["trajectory_digest"]
        )
        parent_result, parent_trajectory = simulate_shared_world(
            asset_root,
            **_context_kwargs(
                context_record=context_record,
                target=parent_target,
                foot_yaw=parent_yaw,
                controller=controller,
                duration=active.simulation_duration_sec,
            ),
        )
        stability = _stability_retained(result.to_dict(), parent_result.to_dict(), active)
        if expected_success:
            passed = bool(
                decision.accepted
                and action_executed
                and _safe(result)
                and result.goal_crossed
                and result.target_error_m is not None
                and result.target_error_m <= active.maximum_target_error_m
                and result.pass_delivery_error_m is not None
                and result.pass_delivery_error_m <= active.maximum_pass_error_m
                and exact
                and stability
            )
        else:
            passed = bool(
                not decision.accepted
                and decision.route == "KNOWN_FINISH_FAILURE_BASIN_FALLBACK"
                and not action_executed
                and _safe(result)
                and exact
            )
        values[context.case_id] = {
            "context": asdict(context),
            "context_hash": context_record["context_hash"],
            "expected": "PRECISE_EXECUTION" if expected_success else "FAILURE_VETO",
            "decision": asdict(decision),
            "candidate_action_executed": action_executed,
            "executed_policy_target_m": list(executed_target),
            "executed_foot_yaw_offset_rad": executed_yaw,
            "result": result.to_dict(),
            "trajectory": record,
            "replay": {"result": replay_result.to_dict(), "trajectory": replay_record},
            "parent": {
                "result": parent_result.to_dict(),
                "trajectory": _save_trajectory(
                    output / f"holdout-{context.case_id}-parent.npz",
                    parent_trajectory,
                ),
            },
            "exact_replay": exact,
            "stability_retained": stability,
            "passed": passed,
        }
    return values


def _resolve_context(
    *,
    context: FinishPortfolioContext,
    policy: DynamicLeadPassPolicy,
    physical_target: tuple[float, float, float],
) -> dict[str, Any]:
    reception_target = policy.reception_target(
        receiver_phase_start_sec=context.receiver_phase_start_sec,
        receiver_lateral_lane_m=context.receiver_lateral_lane_m,
    )
    passer_yaw = policy.passer_world_yaw(target_lateral_m=context.receiver_lateral_lane_m)
    base = three_role_development_kwargs()
    features = prepared_finish_plan_features(
        receiver_lane_m=context.receiver_lateral_lane_m,
        reception_target_x_m=reception_target[0],
        passer_ball_local_xy_m=base["passer_ball_local_xy"],
        ball_ground_friction=float(base["ball_ground_friction"]),
        passer_yaw_rad=passer_yaw,
        passer_stance_offset_xy_m=(0.0, 0.0),
        passer_swing_speed_scale=0.80,
    )
    context_hash = str(hash_json({"context": asdict(context), "features": features}))
    return {
        "context_hash": context_hash,
        "features": list(features),
        "receiver_phase_start_sec": context.receiver_phase_start_sec,
        "receiver_lateral_lane_m": context.receiver_lateral_lane_m,
        "pass_reception_target_m": list(reception_target),
        "passer_yaw_rad": passer_yaw,
        "physical_target_m": list(physical_target),
    }


def _context_kwargs(
    *,
    context_record: dict[str, Any],
    target: tuple[float, float, float],
    foot_yaw: float,
    controller: ContextualFinishTargetGrowthConfig,
    duration: float,
) -> dict[str, Any]:
    request = {
        "receiver_phase_start_sec": context_record["receiver_phase_start_sec"],
        "receiver_lateral_lane_m": context_record["receiver_lateral_lane_m"],
        "pass_reception_target_m": context_record["pass_reception_target_m"],
        "passer_yaw_rad": context_record["passer_yaw_rad"],
    }
    return cast(
        dict[str, Any],
        _simulation_kwargs(
            request=request,
            policy_target=target,
            foot_yaw_offset_rad=foot_yaw,
            duration=duration,
            config=controller,
        ),
    )


def _run_jobs(
    jobs: list[tuple[Path, dict[str, Any]]], workers: int
) -> list[tuple[G1SharedWorldResult, dict[str, np.ndarray]]]:
    if workers == 1:
        return [simulate_shared_world(asset, **kwargs) for asset, kwargs in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(simulate_shared_world, asset, **kwargs) for asset, kwargs in jobs
        ]
        return [future.result() for future in futures]


def _success_sample(
    *,
    context: FinishPortfolioContext,
    context_record: dict[str, Any],
    outcome: dict[str, Any],
    physical_target: tuple[float, float, float],
    control_hash: str,
) -> FinishTargetCalibrationSample:
    if not outcome["passed"]:
        raise ValueError(f"success discovery {context.case_id} did not pass")
    selected = cast(dict[str, Any], outcome["selected"])
    result = cast(dict[str, Any], selected["result"])
    return FinishTargetCalibrationSample(
        context_hash=str(context_record["context_hash"]),
        trajectory_hash=str(selected["trajectory"]["trajectory_digest"]),
        control_envelope_hash=control_hash,
        features=cast(tuple[float, ...], tuple(context_record["features"])),
        requested_physical_target_m=physical_target,
        executed_policy_target_m=cast(
            tuple[float, float, float], tuple(selected["policy_target_m"])
        ),
        executed_foot_yaw_offset_rad=float(selected["foot_yaw_offset_rad"]),
        observed_crossing_m=(
            physical_target[0],
            float(result["goal_crossing_y_m"]),
            float(result["goal_crossing_z_m"]),
        ),
        target_error_m=float(result["target_error_m"]),
        safe=True,
        exact_replay=True,
    )


def _failure_memory(
    *,
    context: FinishPortfolioContext,
    context_record: dict[str, Any],
    outcome: dict[str, Any],
    control_hash: str,
) -> FinishTargetFailureMemory:
    if not outcome["failure_proved"]:
        raise ValueError(f"failure discovery {context.case_id} was not proved")
    error = outcome["selected"]["result"].get("target_error_m")
    return FinishTargetFailureMemory(
        context_hash=str(context_record["context_hash"]),
        search_hash=str(
            hash_json(
                {
                    "context_hash": context_record["context_hash"],
                    "rows": outcome["rows"],
                    "selected_exact_replay": outcome["selected_exact_replay"],
                }
            )
        ),
        control_envelope_hash=control_hash,
        features=cast(tuple[float, ...], tuple(context_record["features"])),
        failure_code="NO_PRECISE_SAFE_ACTION_IN_BOUNDED_SEARCH",
        candidate_count=int(outcome["candidate_count"]),
        safe_candidate_count=int(outcome["safe_candidate_count"]),
        best_safe_target_error_m=None if error is None else float(error),
        exact_replay=True,
    )


def _precise(row: dict[str, Any], config: ContextualFinishPortfolioConfig) -> bool:
    error = row["result"].get("target_error_m")
    return bool(
        row["safe"]
        and row["result"].get("goal_crossed") is True
        and isinstance(error, int | float)
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        and float(error) <= config.maximum_target_error_m
    )


def _stability_retained(
    result: dict[str, Any], parent: dict[str, Any], config: ContextualFinishPortfolioConfig
) -> bool:
    return bool(
        float(result["shooter_min_pelvis_height_m"]) + config.maximum_pelvis_regression_m
        >= float(parent["shooter_min_pelvis_height_m"])
        and float(result["shooter_post_contact_support_foot_slip_m"])
        <= float(parent["shooter_post_contact_support_foot_slip_m"])
        + config.maximum_support_slip_regression_m
        and float(result["shooter_post_contact_support_foot_slip_m"])
        <= config.maximum_absolute_support_slip_m
    )


def _metrics(
    successes: dict[str, Any], failures: dict[str, Any], holdouts: dict[str, Any]
) -> dict[str, Any]:
    success_holdouts = tuple(
        case for case in holdouts.values() if case["expected"] == "PRECISE_EXECUTION"
    )
    errors = [float(case["result"]["target_error_m"]) for case in success_holdouts]
    return {
        "success_discovery_context_count": len(successes),
        "failure_discovery_context_count": len(failures),
        "sealed_holdout_count": len(holdouts),
        "success_discovery_target_errors_m": {
            case_id: case["selected"]["result"]["target_error_m"]
            for case_id, case in successes.items()
        },
        "failure_discovery_best_errors_m": {
            case_id: case["selected"]["result"].get("target_error_m")
            for case_id, case in failures.items()
        },
        "success_holdout_target_errors_m": {
            case_id: case["result"]["target_error_m"]
            for case_id, case in holdouts.items()
            if case["expected"] == "PRECISE_EXECUTION"
        },
        "maximum_success_holdout_target_error_m": max(errors),
        "failure_holdout_veto_rate": sum(
            case["passed"] for case in holdouts.values() if case["expected"] == "FAILURE_VETO"
        )
        / max(1, sum(case["expected"] == "FAILURE_VETO" for case in holdouts.values())),
    }


def validate_contextual_finish_portfolio(path: Path) -> dict[str, Any]:
    """Validate report, actor and every raw trajectory content binding."""

    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S204 report must be an object")
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = report_path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        active = _config_from_dict(cast(dict[str, Any], request["config"]))
        controller = _control_config_from_dict(cast(dict[str, Any], request["control_config"]))
        if (
            expected_hash != hash_json(payload)
            or payload.get("implementation_hash") != _implementation_hash()
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or request.get("implementation_hash") != _implementation_hash()
            or request.get("config_hash") != active.config_hash
            or request.get("control_envelope_hash") != _control_envelope_hash(controller)
            or payload.get("source_s203_hash") != request.get("source_s203_hash")
            or request.get("physics_authority") != "CPU_MUJOCO"
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or request.get("pixels_used_for_scoring") is not False
        ):
            raise ValueError("S204 report integrity changed")
        source_path = _source_s203_path(request, report_path)
        source = validate_contextual_finish_target_growth(source_path)
        if source.get("report_hash") != payload.get("source_s203_hash") or hash_bytes(
            source_path.read_bytes()
        ) != request.get("source_s203_file_hash"):
            raise ValueError("S204 source S203 binding changed")
        resolved = cast(dict[str, Any], request["resolved_contexts"])
        expected_contexts = _all_contexts(active)
        if set(resolved) != {context.case_id for context in expected_contexts}:
            raise ValueError("S204 resolved context partition changed")
        for context in expected_contexts:
            _validate_resolved_context(
                context=context,
                value=cast(dict[str, Any], resolved[context.case_id]),
                physical_target=cast(
                    tuple[float, float, float], tuple(request["physical_target_m"])
                ),
            )
        successes = cast(dict[str, Any], payload["success_discovery"])
        failures = cast(dict[str, Any], payload["failure_discovery"])
        if set(successes) != {context.case_id for context in active.success_discovery} or set(
            failures
        ) != {context.case_id for context in active.failure_discovery}:
            raise ValueError("S204 discovery partition changed")
        for context in active.success_discovery:
            _validate_search_case(
                root=report_path.parent,
                context=context,
                context_record=cast(dict[str, Any], resolved[context.case_id]),
                case=cast(dict[str, Any], successes[context.case_id]),
                y_candidates=active.success_policy_target_y_candidates_m,
                yaw_candidates=active.success_foot_yaw_candidates_rad,
                request=request,
                active=active,
                expected_success=True,
            )
        for context in active.failure_discovery:
            _validate_search_case(
                root=report_path.parent,
                context=context,
                context_record=cast(dict[str, Any], resolved[context.case_id]),
                case=cast(dict[str, Any], failures[context.case_id]),
                y_candidates=active.failure_policy_target_y_candidates_m,
                yaw_candidates=active.failure_foot_yaw_candidates_rad,
                request=request,
                active=active,
                expected_success=False,
            )
        actor_record = cast(dict[str, Any], payload["actor"])
        actor_path = report_path.parent / str(actor_record["file"])
        actor = load_contextual_finish_target_actor(actor_path)
        if (
            hash_bytes(actor_path.read_bytes()) != actor_record.get("file_hash")
            or actor.actor_hash != actor_record.get("actor_hash")
            or actor.body_hash != request.get("body_hash")
            or actor.kick_prior_hash != request.get("kick_prior_hash")
            or actor.roster_hash != request.get("roster_hash")
            or actor.finisher_self_model_hash != request.get("finisher_self_model_hash")
            or actor.control_envelope_hash != request.get("control_envelope_hash")
            or actor.source_evidence_hashes != (str(payload.get("source_s203_hash")),)
            or actor.distinct_context_count != 6
            or actor.distinct_trajectory_count != 6
            or len(actor.failure_memories) != 2
            or not actor.evidence_ready
            or actor.nearest_sample_count != 1
            or actor.failure_exclusion_distance != active.failure_exclusion_distance
            or actor_record.get("evidence_ready") is not actor.evidence_ready
            or actor_record.get("failure_memory_count") != len(actor.failure_memories)
        ):
            raise ValueError("S204 actor binding changed")
        samples = {sample.context_hash: sample for sample in actor.samples}
        for context in active.success_discovery:
            case = cast(dict[str, Any], successes[context.case_id])
            context_record = cast(dict[str, Any], resolved[context.case_id])
            sample = samples.get(str(context_record["context_hash"]))
            selected = cast(dict[str, Any], case["selected"])
            result = cast(dict[str, Any], selected["result"])
            if sample is None or (
                sample.trajectory_hash != selected["trajectory"]["trajectory_digest"]
                or sample.features != tuple(context_record["features"])
                or list(sample.requested_physical_target_m) != request["physical_target_m"]
                or list(sample.executed_policy_target_m) != selected["policy_target_m"]
                or sample.executed_foot_yaw_offset_rad != selected["foot_yaw_offset_rad"]
                or list(sample.observed_crossing_m)
                != [
                    request["physical_target_m"][0],
                    result["goal_crossing_y_m"],
                    result["goal_crossing_z_m"],
                ]
                or sample.target_error_m != result["target_error_m"]
            ):
                raise ValueError("S204 success memory binding changed")
        memories = {memory.context_hash: memory for memory in actor.failure_memories}
        for context in active.failure_discovery:
            case = cast(dict[str, Any], failures[context.case_id])
            context_record = cast(dict[str, Any], resolved[context.case_id])
            memory = memories.get(str(context_record["context_hash"]))
            expected_search_hash = hash_json(
                {
                    "context_hash": context_record["context_hash"],
                    "rows": case["rows"],
                    "selected_exact_replay": case["selected_exact_replay"],
                }
            )
            expected_error = case["selected"]["result"].get("target_error_m")
            if memory is None or (
                memory.search_hash != expected_search_hash
                or memory.features != tuple(context_record["features"])
                or memory.candidate_count != case["candidate_count"]
                or memory.safe_candidate_count != case["safe_candidate_count"]
                or memory.best_safe_target_error_m != expected_error
            ):
                raise ValueError("S204 failure memory binding changed")
        holdouts = cast(dict[str, Any], payload["holdouts"])
        if set(holdouts) != {
            context.case_id for context in (*active.success_holdouts, *active.failure_holdouts)
        }:
            raise ValueError("S204 holdout partition changed")
        success_holdout_ids = {context.case_id for context in active.success_holdouts}
        for context in (*active.success_holdouts, *active.failure_holdouts):
            case = cast(dict[str, Any], holdouts[context.case_id])
            context_record = cast(dict[str, Any], resolved[context.case_id])
            for record in (
                case["trajectory"],
                case["replay"]["trajectory"],
                case["parent"]["trajectory"],
            ):
                _validate_trajectory(report_path.parent, record)
            decision = actor.decide(
                cast(tuple[float, ...], tuple(context_record["features"])),
                cast(tuple[float, float, float], tuple(context_record["physical_target_m"])),
            )
            expected_success = context.case_id in success_holdout_ids
            result = cast(dict[str, Any], case["result"])
            parent = cast(dict[str, Any], case["parent"]["result"])
            exact = bool(
                result == case["replay"]["result"]
                and case["trajectory"]["trajectory_digest"]
                == case["replay"]["trajectory"]["trajectory_digest"]
            )
            stability = _stability_retained(result, parent, active)
            passed = (
                bool(
                    decision.accepted
                    and case["candidate_action_executed"]
                    and _safe_result_dict(result)
                    and result.get("goal_crossed") is True
                    and isinstance(result.get("target_error_m"), int | float)
                    and float(result["target_error_m"]) <= active.maximum_target_error_m
                    and isinstance(result.get("pass_delivery_error_m"), int | float)
                    and float(result["pass_delivery_error_m"]) <= active.maximum_pass_error_m
                    and exact
                    and stability
                )
                if expected_success
                else bool(
                    not decision.accepted
                    and decision.route == "KNOWN_FINISH_FAILURE_BASIN_FALLBACK"
                    and not case["candidate_action_executed"]
                    and _safe_result_dict(result)
                    and exact
                )
            )
            expected_target = (
                decision.policy_target_m
                if expected_success and decision.accepted
                else tuple(request["parent_policy_target_m"])
            )
            expected_yaw = (
                decision.foot_yaw_offset_rad
                if expected_success and decision.accepted
                else request["parent_foot_yaw_offset_rad"]
            )
            if (
                case.get("context") != asdict(context)
                or case.get("context_hash") != context_record["context_hash"]
                or hash_json(case.get("decision")) != hash_json(asdict(decision))
                or case.get("expected")
                != ("PRECISE_EXECUTION" if expected_success else "FAILURE_VETO")
                or case.get("executed_policy_target_m")
                != list(cast(tuple[Any, ...], expected_target))
                or case.get("executed_foot_yaw_offset_rad") != expected_yaw
                or case.get("exact_replay") is not exact
                or case.get("stability_retained") is not stability
                or case.get("passed") is not passed
            ):
                raise ValueError("S204 holdout derivation changed")
        discovery_exact = all(
            case["selected_exact_replay"] for case in (*successes.values(), *failures.values())
        )
        holdout_passed = all(case["passed"] for case in holdouts.values())
        parent_retention = all(case["stability_retained"] for case in successes.values()) and all(
            case["stability_retained"] for case in holdouts.values()
        )
        candidate = _candidate_from_dict(cast(dict[str, Any], payload["backend_candidate"]))
        if (
            candidate.backend is not RoleOptionBackend.CONTEXTUAL_FINISH_TARGET
            or candidate.option is not PhysicalSoccerOption.SHOOT
            or candidate.artifact_hash != actor.actor_hash
            or candidate.evidence_hash != payload.get("source_s203_hash")
            or candidate.distinct_context_count != actor.distinct_context_count
            or candidate.distinct_trajectory_count != actor.distinct_trajectory_count
            or candidate.strict_replay is not discovery_exact
            or candidate.holdout_passed is not holdout_passed
            or candidate.parent_retention_passed is not parent_retention
            or candidate.candidate_hash != payload.get("backend_candidate_hash")
        ):
            raise ValueError("S204 backend candidate binding changed")
        route = _route_from_dict(cast(dict[str, Any], payload["route"]))
        expected_route_accepted = candidate.evidence_ready
        if (
            route.agent_id != "red.finisher"
            or route.cell_hash != request.get("finisher_cell_hash")
            or route.option is not PhysicalSoccerOption.SHOOT
            or route.accepted is not expected_route_accepted
            or route.selected_backend
            is not (RoleOptionBackend.CONTEXTUAL_FINISH_TARGET if expected_route_accepted else None)
            or route.candidate_hash
            != (candidate.candidate_hash if expected_route_accepted else None)
            or route.reason
            != (
                "ROLE_AND_EVIDENCE_QUALIFIED"
                if expected_route_accepted
                else "NO_ROLE_QUALIFIED_BACKEND"
            )
            or route.route_hash != payload.get("route_hash")
        ):
            raise ValueError("S204 role route binding changed")
        gates = {
            "source_single_context_seed_bound": True,
            "six_distinct_success_contexts": actor.distinct_context_count == 6,
            "multiple_physical_success_lanes": len(
                {context.receiver_lateral_lane_m for context in active.success_discovery}
            )
            >= 2,
            "all_success_discovery_precise_safe": all(
                case["passed"] for case in successes.values()
            ),
            "bounded_failure_searches_proved": all(
                case["failure_proved"] for case in failures.values()
            ),
            "failure_memories_bound": len(actor.failure_memories) == 2,
            "actor_evidence_ready": actor.evidence_ready,
            "sealed_success_holdouts_passed": all(
                holdouts[context.case_id]["passed"] for context in active.success_holdouts
            ),
            "sealed_failure_holdouts_vetoed": all(
                holdouts[context.case_id]["passed"] for context in active.failure_holdouts
            ),
            "strict_replay_complete": discovery_exact
            and all(case["exact_replay"] for case in holdouts.values()),
            "parent_stability_retained": parent_retention,
            "backend_evidence_ready": candidate.evidence_ready,
            "role_qualified_route_accepted": route.accepted
            and route.selected_backend is RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        }
        passed = all(gates.values())
        if (
            payload.get("metrics") != _metrics(successes, failures, holdouts)
            or payload.get("gates") != gates
            or payload.get("passed") is not passed
            or payload.get("status")
            != (
                "PASS_CONTEXTUAL_FINISH_PORTFOLIO_AND_FAILURE_VETO"
                if passed
                else "REJECTED_CONTEXTUAL_FINISH_PORTFOLIO"
            )
            or payload.get("promotion_eligible") is not False
            or payload.get("evidence_boundary", {}).get("partition")
            != "DEVELOPMENT_WITH_SEALED_HOLDOUT"
            or payload.get("evidence_boundary", {}).get("plastic_agent_id") != "red.finisher"
            or payload.get("evidence_boundary", {}).get("frozen_role_count") != 5
            or payload.get("evidence_boundary", {}).get("activation_ceiling") != "SIM_ONLY"
            or payload.get("evidence_boundary", {}).get("hardware_command_sent") is not False
            or payload.get("evidence_boundary", {}).get("pixels_used_for_scoring") is not False
        ):
            raise ValueError("S204 derived gates or authority changed")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return payload


def _validate_resolved_context(
    *,
    context: FinishPortfolioContext,
    value: dict[str, Any],
    physical_target: tuple[float, float, float],
) -> None:
    features = value.get("features")
    if not isinstance(features, list) or value.get("context_hash") != hash_json(
        {"context": asdict(context), "features": tuple(features)}
    ):
        raise ValueError("S204 context commitment changed")
    if (
        value.get("receiver_phase_start_sec") != context.receiver_phase_start_sec
        or value.get("receiver_lateral_lane_m") != context.receiver_lateral_lane_m
        or value.get("physical_target_m") != list(physical_target)
    ):
        raise ValueError("S204 resolved context changed")


def _validate_search_case(
    *,
    root: Path,
    context: FinishPortfolioContext,
    context_record: dict[str, Any],
    case: dict[str, Any],
    y_candidates: tuple[float, ...],
    yaw_candidates: tuple[float, ...],
    request: dict[str, Any],
    active: ContextualFinishPortfolioConfig,
    expected_success: bool,
) -> None:
    rows = case.get("rows")
    if not isinstance(rows, list) or len(rows) != len(y_candidates) * len(yaw_candidates):
        raise ValueError("S204 search rows changed")
    expected_actions = tuple((y, yaw) for y in y_candidates for yaw in yaw_candidates)
    for index, (row, (target_y, foot_yaw)) in enumerate(zip(rows, expected_actions, strict=True)):
        _validate_trajectory(root, row["trajectory"])
        if (
            row.get("candidate_index") != index
            or row.get("policy_target_m")
            != [request["physical_target_m"][0], target_y, active.policy_target_z_m]
            or row.get("foot_yaw_offset_rad") != foot_yaw
            or row.get("safe") is not _safe_result_dict(cast(dict[str, Any], row["result"]))
        ):
            raise ValueError("S204 candidate derivation changed")
    selected = min(rows, key=_selection_key)
    replay = cast(dict[str, Any], case["selected_replay"])
    parent = cast(dict[str, Any], case["parent"])
    _validate_trajectory(root, replay["trajectory"])
    _validate_trajectory(root, parent["trajectory"])
    exact = bool(
        selected["result"] == replay["result"]
        and selected["trajectory"]["trajectory_digest"] == replay["trajectory"]["trajectory_digest"]
    )
    stability = _stability_retained(selected["result"], parent["result"], active)
    precise_rows = tuple(row for row in rows if _precise(row, active))
    selected_passed = bool(
        _precise(selected, active)
        and exact
        and isinstance(selected["result"].get("pass_delivery_error_m"), int | float)
        and float(selected["result"]["pass_delivery_error_m"]) <= active.maximum_pass_error_m
        and stability
    )
    failure_proved = bool(
        not precise_rows and selected["safe"] and exact and len(rows) == len(expected_actions)
    )
    if (
        case.get("context") != asdict(context)
        or case.get("context_hash") != context_record["context_hash"]
        or case.get("candidate_count") != len(rows)
        or case.get("safe_candidate_count") != sum(bool(row["safe"]) for row in rows)
        or case.get("precise_safe_candidate_count") != len(precise_rows)
        or case.get("selected") != selected
        or case.get("selected_exact_replay") is not exact
        or case.get("stability_retained") is not stability
        or case.get("failure_proved") is not failure_proved
        or case.get("passed") is not (selected_passed if expected_success else failure_proved)
        or case.get("parent_action", {}).get("policy_target_m") != request["parent_policy_target_m"]
        or case.get("parent_action", {}).get("foot_yaw_offset_rad")
        != request["parent_foot_yaw_offset_rad"]
        or parent.get("safe") is not _safe_result_dict(cast(dict[str, Any], parent["result"]))
    ):
        raise ValueError("S204 search case derivation changed")


def _safe_result_dict(result: dict[str, Any]) -> bool:
    return bool(
        result.get("finite_state") is True
        and result.get("actuator_saturation") is False
        and result.get("torque_limit_violation") is False
        and result.get("joint_limit_violation") is False
        and result.get("passer_joint_limit_violation") is False
        and result.get("shooter_joint_limit_violation") is False
        and result.get("goalkeeper_joint_limit_violation") is False
        and result.get("passer_post_kick_fall") is False
        and result.get("shooter_post_kick_fall") is False
        and result.get("robot_robot_contact_count") == 0
        and float(result.get("passer_min_pelvis_height_m", -math.inf)) >= 0.60
        and float(result.get("shooter_min_pelvis_height_m", -math.inf)) >= 0.60
        and isinstance(result.get("goalkeeper_min_pelvis_height_m"), int | float)
        and float(result["goalkeeper_min_pelvis_height_m"]) >= 0.60
    )


def _candidate_from_dict(value: dict[str, Any]) -> RoleOptionBackendCandidate:
    fields = dict(value)
    claimed_ready = fields.pop("evidence_ready", None)
    candidate = RoleOptionBackendCandidate(
        **{
            **fields,
            "backend": RoleOptionBackend(fields["backend"]),
            "option": PhysicalSoccerOption(fields["option"]),
        }
    )
    if claimed_ready is not candidate.evidence_ready:
        raise ValueError("S204 candidate readiness changed")
    return candidate


def _route_from_dict(value: dict[str, Any]) -> RoleOptionBackendRoute:
    return RoleOptionBackendRoute(
        **{
            **value,
            "option": PhysicalSoccerOption(value["option"]),
            "selected_backend": (
                None
                if value["selected_backend"] is None
                else RoleOptionBackend(value["selected_backend"])
            ),
        }
    )


def _source_s203_path(request: dict[str, Any], report_path: Path) -> Path:
    source_hash = request.get("source_s203_file_hash")
    candidates = tuple(
        report_path.parents[1].glob(
            "s203-contextual-finish-target-*/contextual-finish-target-growth.json"
        )
    )
    for candidate in candidates:
        if hash_bytes(candidate.read_bytes()) == source_hash:
            return candidate
    raise ValueError("S204 source S203 file is unavailable")


def _config_from_dict(value: dict[str, Any]) -> ContextualFinishPortfolioConfig:
    def contexts(name: str) -> tuple[FinishPortfolioContext, ...]:
        return tuple(FinishPortfolioContext(**item) for item in value[name])

    return ContextualFinishPortfolioConfig(
        **{
            **value,
            "success_discovery": contexts("success_discovery"),
            "failure_discovery": contexts("failure_discovery"),
            "success_holdouts": contexts("success_holdouts"),
            "failure_holdouts": contexts("failure_holdouts"),
            "success_policy_target_y_candidates_m": tuple(
                value["success_policy_target_y_candidates_m"]
            ),
            "success_foot_yaw_candidates_rad": tuple(value["success_foot_yaw_candidates_rad"]),
            "failure_policy_target_y_candidates_m": tuple(
                value["failure_policy_target_y_candidates_m"]
            ),
            "failure_foot_yaw_candidates_rad": tuple(value["failure_foot_yaw_candidates_rad"]),
        }
    )


def _control_config_from_dict(
    value: dict[str, Any],
) -> ContextualFinishTargetGrowthConfig:
    return ContextualFinishTargetGrowthConfig(
        **{
            **value,
            "policy_target_y_candidates_m": tuple(value["policy_target_y_candidates_m"]),
            "foot_yaw_offset_candidates_rad": tuple(value["foot_yaw_offset_candidates_rad"]),
        }
    )


def _all_contexts(config: ContextualFinishPortfolioConfig) -> tuple[FinishPortfolioContext, ...]:
    return (
        *config.success_discovery,
        *config.failure_discovery,
        *config.success_holdouts,
        *config.failure_holdouts,
    )


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
        Path(__file__).parents[1] / "growth" / "contextual_finish_target.py",
        Path(__file__).parents[1] / "growth" / "role_option_backend.py",
        Path(__file__).parent / "contextual_finish_target_growth.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_output(path: Path, checkout: Path) -> Path:
    output = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("S204 evidence output must be new and external")
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
    parser.add_argument("--source-s203", type=Path, required=True)
    parser.add_argument("--source-lead-pass", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_contextual_finish_portfolio(
        asset_root=args.asset_root,
        source_s203_path=args.source_s203,
        source_lead_pass_dir=args.source_lead_pass,
        source_checkout=args.source_checkout,
        output_dir=args.output,
        workers=args.workers,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContextualFinishPortfolioConfig",
    "FinishPortfolioContext",
    "run_contextual_finish_portfolio",
    "validate_contextual_finish_portfolio",
]
