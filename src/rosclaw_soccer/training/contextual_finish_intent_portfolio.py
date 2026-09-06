"""S206 grow a multi-context four-dimensional finish-intent expert.

S206 combines passing S204 discovery experts with the S205 failure repair,
learns a second repaired basin, and evaluates the frozen actor on preregistered
new contexts.  All actions are high-level inputs to a frozen whole-body prior;
the stage is CPU MuJoCo and SIM_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.bounded_active_search import (
    BoundedActiveSearchPlan,
    BoundedSearchDimension,
)
from rosclaw_soccer.growth.contextual_finish_intent import (
    DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE,
    ContextualFinishIntentAction,
    ContextualFinishIntentActor,
    ContextualFinishIntentSample,
    contextual_finish_intent_features,
    load_contextual_finish_intent_actor,
    save_contextual_finish_intent_actor,
)
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
    RoleOptionBackendRoute,
)
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.contextual_finish_portfolio import (
    ContextualFinishPortfolioConfig,
    FinishPortfolioContext,
    _control_config_from_dict,
    _load_lead_pass,
    _resolve_context,
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
    _save_trajectory,
    _validate_trajectory,
)
from rosclaw_soccer.training.extended_finish_intent_repair import (
    ExtendedFinishIntentRepairConfig,
    _candidate_kwargs,
    _candidate_row,
    _refinement_seed_key,
    _selection_key,
    _validate_candidate_row,
    validate_extended_finish_intent_repair,
)
from rosclaw_soccer.training.independent_team_growth import (
    build_independent_three_vs_three_fixture,
)


@dataclass(frozen=True)
class ContextualFinishIntentPortfolioConfig:
    """Fixed B-basin repair plan and fresh A/B/OOD evaluation contexts."""

    repair_source_case_id: str = "basin-b-sealed"
    repair_policy_target_y_bounds_m: tuple[float, float] = (0.38, 0.48)
    repair_foot_yaw_bounds_rad: tuple[float, float] = (0.035, 0.075)
    repair_stance_offset_y_bounds_m: tuple[float, float] = (-0.08, 0.02)
    repair_foot_pitch_bounds_rad: tuple[float, float] = (-0.02, 0.08)
    repair_warm_start: tuple[float, float, float, float] = (0.43, 0.055, -0.04, 0.01)
    coarse_candidate_count: int = 64
    coarse_sequence_skip: int = 256
    refinement_candidate_count: int = 32
    refinement_sequence_skip: int = 512
    refinement_radius_fraction: float = 0.10
    maximum_target_error_m: float = 0.10
    maximum_pass_error_m: float = 0.05
    minimum_repair_improvement_m: float = 0.10
    simulation_duration_sec: float = 10.0
    success_holdouts: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("s206-a-holdout", 1.9000, 0.07886),
        FinishPortfolioContext("s206-b-holdout", 1.9200, 0.09927),
    )
    rejection_holdouts: tuple[FinishPortfolioContext, ...] = (
        FinishPortfolioContext("s206-phase-ood", 1.9005, 0.07882),
        FinishPortfolioContext("s206-gap-ood", 1.9120, 0.09050),
    )
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.contextual_finish_intent_portfolio_config.v1"

    def __post_init__(self) -> None:
        bounds = self.repair_bounds
        contexts = (*self.success_holdouts, *self.rejection_holdouts)
        if (
            self.repair_source_case_id != "basin-b-sealed"
            or any(bound[0] >= bound[1] for bound in bounds)
            or len(self.repair_warm_start) != 4
            or any(
                not bound[0] <= value <= bound[1]
                for value, bound in zip(self.repair_warm_start, bounds, strict=True)
            )
            or self.coarse_candidate_count != 64
            or self.coarse_sequence_skip != 256
            or self.refinement_candidate_count != 32
            or self.refinement_sequence_skip != 512
            or self.refinement_radius_fraction != 0.10
            or not 0.05 <= self.maximum_target_error_m <= 0.10
            or not 0.01 <= self.maximum_pass_error_m <= 0.05
            or not 0.05 <= self.minimum_repair_improvement_m <= 0.25
            or not 8.0 <= self.simulation_duration_sec <= 15.0
            or len(self.success_holdouts) != 2
            or len(self.rejection_holdouts) != 2
            or len({context.case_id for context in contexts}) != 4
            or len(
                {
                    (context.receiver_phase_start_sec, context.receiver_lateral_lane_m)
                    for context in contexts
                }
            )
            != 4
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.contextual_finish_intent_portfolio_config.v1"
        ):
            raise ValueError("contextual finish intent portfolio config is invalid")

    @property
    def repair_bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            self.repair_policy_target_y_bounds_m,
            self.repair_foot_yaw_bounds_rad,
            self.repair_stance_offset_y_bounds_m,
            self.repair_foot_pitch_bounds_rad,
        )

    def _plan(
        self, *, candidate_count: int, sequence_skip: int, radius: float
    ) -> BoundedActiveSearchPlan:
        return BoundedActiveSearchPlan(
            dimensions=tuple(
                BoundedSearchDimension(name, lower, upper)
                for name, (lower, upper) in zip(
                    (
                        "policy_target_y_m",
                        "foot_yaw_offset_rad",
                        "stance_offset_y_m",
                        "foot_pitch_offset_rad",
                    ),
                    self.repair_bounds,
                    strict=True,
                )
            ),
            global_candidate_count=64,
            local_candidate_count=candidate_count,
            sequence_skip=sequence_skip,
            local_radius_fraction=radius,
        )

    @property
    def coarse_plan(self) -> BoundedActiveSearchPlan:
        return self._plan(
            candidate_count=self.coarse_candidate_count,
            sequence_skip=self.coarse_sequence_skip,
            radius=0.50,
        )

    @property
    def refinement_plan(self) -> BoundedActiveSearchPlan:
        return self._plan(
            candidate_count=self.refinement_candidate_count,
            sequence_skip=self.refinement_sequence_skip,
            radius=self.refinement_radius_fraction,
        )

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_contextual_finish_intent_portfolio(
    *,
    asset_root: Path,
    source_s204_path: Path,
    source_s205_path: Path,
    source_lead_pass_dir: Path,
    source_checkout: Path,
    output_dir: Path,
    config: ContextualFinishIntentPortfolioConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Repair a second basin, freeze an actor, then run fresh holdouts."""

    if not 1 <= workers <= 4:
        raise ValueError("S206 workers must be in [1, 4]")
    active = config or ContextualFinishIntentPortfolioConfig()
    source204_path = source_s204_path.expanduser().resolve()
    source205_path = source_s205_path.expanduser().resolve()
    source204 = validate_contextual_finish_portfolio(source204_path)
    source205 = validate_extended_finish_intent_repair(source205_path)
    request204_path = source204_path.parent / "request.json"
    request204 = json.loads(request204_path.read_text(encoding="utf-8"))
    if (
        source204.get("status") != "REJECTED_CONTEXTUAL_FINISH_PORTFOLIO"
        or source205.get("status") != "PASS_EXTENDED_FINISH_INTENT_REPAIR"
        or source205.get("source_s204_hash") != source204.get("report_hash")
        or source205.get("promotion_eligible") is not False
    ):
        raise ValueError("S206 requires bound S204 rejection and passing S205 repair")
    policy, policy_source = _load_lead_pass(source_lead_pass_dir)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(asset_root)
    finisher = next(cell for cell in fixture.cells if cell.agent_id == "red.finisher")
    control_hash = str(request204["control_envelope_hash"])
    controller = _control_config_from_dict(cast(dict[str, Any], request204["control_config"]))
    portfolio_config = _portfolio_config_from_dict(cast(dict[str, Any], request204["config"]))
    physical_target = cast(tuple[float, float, float], tuple(request204["physical_target_m"]))
    parent_target = cast(tuple[float, float, float], tuple(request204["parent_policy_target_m"]))
    parent_yaw = float(request204["parent_foot_yaw_offset_rad"])
    if (
        request204.get("body_hash") != qualification.body_hash
        or request204.get("kick_prior_hash") != qualification.kick_prior_hash
        or request204.get("roster_hash") != fixture.roster.roster_hash
        or request204.get("finisher_self_model_hash") != finisher.self_model.self_model_hash
        or request204.get("source_lead_pass_policy_hash") != policy.artifact_hash
    ):
        raise ValueError("S206 physical lineage changed")
    output = _new_output(output_dir, source_checkout)
    all_holdouts = (*active.success_holdouts, *active.rejection_holdouts)
    resolved_holdouts = {
        context.case_id: _resolve_context(
            context=context,
            policy=policy,
            physical_target=physical_target,
        )
        for context in all_holdouts
    }
    prior_context_hashes = {
        record["context_hash"] for record in request204["resolved_contexts"].values()
    }
    holdout_context_hashes = {record["context_hash"] for record in resolved_holdouts.values()}
    if prior_context_hashes.intersection(holdout_context_hashes):
        raise ValueError("S206 holdouts overlap prior development contexts")
    repair_source = cast(dict[str, Any], source204["holdouts"][active.repair_source_case_id])
    repair_context = cast(
        dict[str, Any],
        request204["resolved_contexts"][active.repair_source_case_id],
    )
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_intent_portfolio_request.v1",
        "source_commit": _git_head(source_checkout),
        "source_s204_hash": source204["report_hash"],
        "source_s204_file_hash": hash_bytes(source204_path.read_bytes()),
        "source_s204_request_hash": hash_bytes(request204_path.read_bytes()),
        "source_s205_hash": source205["report_hash"],
        "source_s205_file_hash": hash_bytes(source205_path.read_bytes()),
        "source_lead_pass_evidence_hash": policy_source["evidence_hash"],
        "source_lead_pass_policy_hash": policy.artifact_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "fixture_hash": fixture.fixture_hash,
        "roster_hash": fixture.roster.roster_hash,
        "finisher_cell_hash": finisher.cell_hash,
        "finisher_self_model_hash": finisher.self_model.self_model_hash,
        "control_config": request204["control_config"],
        "control_envelope_hash": control_hash,
        "physical_target_m": list(physical_target),
        "parent_policy_target_m": list(parent_target),
        "parent_foot_yaw_offset_rad": parent_yaw,
        "legacy_discovery_stance_offset_y_m": -0.06,
        "legacy_discovery_foot_pitch_offset_rad": 0.01,
        "repair_source_context": repair_context,
        "repair_source_context_hash": repair_source["context_hash"],
        "repair_source_trajectory_hash": repair_source["trajectory"]["trajectory_digest"],
        "repair_source_target_error_m": repair_source["result"]["target_error_m"],
        "config": asdict(active),
        "config_hash": active.config_hash,
        "coarse_plan": asdict(active.coarse_plan),
        "coarse_plan_hash": active.coarse_plan.plan_hash,
        "refinement_plan": asdict(active.refinement_plan),
        "refinement_plan_hash": active.refinement_plan.plan_hash,
        "resolved_holdouts": resolved_holdouts,
        "holdout_partition_committed_before_physics": True,
        "feature_scale": list(DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE),
        "maximum_support_distance": 0.35,
        "implementation_hash": _implementation_hash(),
        "runtime": _runtime_manifest(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "direct_joint_torque_output": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)

    repair_criteria = ExtendedFinishIntentRepairConfig()
    baseline_values = (
        float(repair_source["executed_policy_target_m"][1]),
        float(repair_source["executed_foot_yaw_offset_rad"]),
        -0.06,
        0.01,
    )
    baseline_result, baseline_trajectory = simulate_shared_world(
        asset_root,
        **_candidate_kwargs(
            context=repair_context,
            controller=controller,
            duration=active.simulation_duration_sec,
            physical_target=physical_target,
            values=baseline_values,
        ),
    )
    baseline_record = _save_trajectory(output / "repair-source-replay.npz", baseline_trajectory)
    baseline_exact = bool(
        hash_json(baseline_result.to_dict()) == hash_json(repair_source["result"])
        and baseline_record["trajectory_digest"] == repair_source["trajectory"]["trajectory_digest"]
    )
    coarse_planned = active.coarse_plan.local_candidates(active.repair_warm_start)
    coarse_rows = _evaluate_candidates(
        asset_root=asset_root,
        output=output,
        context=repair_context,
        controller=controller,
        physical_target=physical_target,
        parent_result=cast(dict[str, Any], repair_source["parent"]["result"]),
        portfolio_config=portfolio_config,
        criteria=repair_criteria,
        candidates=coarse_planned,
        search_stage="COARSE",
        duration=active.simulation_duration_sec,
        workers=workers,
    )
    refinement_seed = min(coarse_rows, key=_refinement_seed_key)
    refinement_center = cast(tuple[float, ...], tuple(refinement_seed["action_values"]))
    refinement_planned = active.refinement_plan.local_candidates(refinement_center)
    refinement_rows = _evaluate_candidates(
        asset_root=asset_root,
        output=output,
        context=repair_context,
        controller=controller,
        physical_target=physical_target,
        parent_result=cast(dict[str, Any], repair_source["parent"]["result"]),
        portfolio_config=portfolio_config,
        criteria=repair_criteria,
        candidates=refinement_planned,
        search_stage="REFINEMENT",
        duration=active.simulation_duration_sec,
        workers=workers,
    )
    repair_rows = [*coarse_rows, *refinement_rows]
    selected = min(repair_rows, key=_selection_key)
    selected_values = cast(tuple[float, ...], tuple(selected["action_values"]))
    replay_result, replay_trajectory = simulate_shared_world(
        asset_root,
        **_candidate_kwargs(
            context=repair_context,
            controller=controller,
            duration=active.simulation_duration_sec,
            physical_target=physical_target,
            values=selected_values,
        ),
    )
    replay_record = _save_trajectory(
        output / "repair-selected-independent-replay.npz", replay_trajectory
    )
    selected_exact = bool(
        selected["result"] == replay_result.to_dict()
        and selected["trajectory"]["trajectory_digest"] == replay_record["trajectory_digest"]
    )
    repair_improvement = float(repair_source["result"]["target_error_m"]) - float(
        selected["result"]["target_error_m"]
    )
    repair_discovery_body = {
        "source_context_hash": repair_source["context_hash"],
        "coarse_plan_hash": active.coarse_plan.plan_hash,
        "refinement_plan_hash": active.refinement_plan.plan_hash,
        "rows": repair_rows,
        "selected_candidate_hash": selected["candidate_hash"],
        "selected_replay_trajectory_hash": replay_record["trajectory_digest"],
        "selected_exact_replay": selected_exact,
    }
    repair_discovery_hash = str(hash_json(repair_discovery_body))
    samples = _build_samples(
        source204=source204,
        source205=source205,
        request204=request204,
        physical_target=physical_target,
        control_hash=control_hash,
        repair_discovery_hash=repair_discovery_hash,
        repair_source=repair_source,
        repair_context=repair_context,
        selected=selected,
        selected_exact=selected_exact,
    )
    source_hashes = (
        str(source204["report_hash"]),
        str(source205["report_hash"]),
        repair_discovery_hash,
    )
    actor = ContextualFinishIntentActor(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        roster_hash=fixture.roster.roster_hash,
        finisher_self_model_hash=finisher.self_model.self_model_hash,
        control_envelope_hash=control_hash,
        source_evidence_hashes=source_hashes,
        feature_scale=DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE,
        samples=samples,
    )
    actor_path = output / "contextual-finish-intent-actor.json"
    save_contextual_finish_intent_actor(actor, actor_path)
    holdouts = _run_holdouts(
        asset_root=asset_root,
        output=output,
        actor=actor,
        active=active,
        controller=controller,
        portfolio_config=portfolio_config,
        resolved=resolved_holdouts,
        physical_target=physical_target,
        parent_target=parent_target,
        parent_yaw=parent_yaw,
    )
    holdout_passed = all(row["passed"] for row in holdouts.values())
    strict_replay = bool(
        baseline_exact and selected_exact and all(row["exact_replay"] for row in holdouts.values())
    )
    parent_retention = bool(
        selected["stability_retained"]
        and all(row["stability_retained"] for row in holdouts.values())
    )
    evidence_hash = str(hash_json({"sources": source_hashes}))
    candidate = RoleOptionBackendCandidate(
        backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        option=PhysicalSoccerOption.SHOOT,
        artifact_hash=actor.actor_hash,
        evidence_hash=evidence_hash,
        distinct_context_count=actor.distinct_context_count,
        distinct_trajectory_count=actor.distinct_trajectory_count,
        strict_replay=strict_replay,
        holdout_passed=holdout_passed,
        parent_retention_passed=parent_retention,
    )
    route = _route_for_candidate(candidate, finisher.cell_hash)
    success_ids = {context.case_id for context in active.success_holdouts}
    rejection_ids = {context.case_id for context in active.rejection_holdouts}
    gates = {
        "source_lineage_bound": True,
        "source_failure_exactly_replayed": baseline_exact,
        "second_basin_precise_safe": bool(selected["eligible"]),
        "second_basin_minimum_improvement_met": repair_improvement
        >= active.minimum_repair_improvement_m,
        "second_basin_independent_exact_replay": selected_exact,
        "eight_distinct_physical_experts": actor.distinct_context_count == 8
        and actor.distinct_trajectory_count == 8,
        "actor_evidence_ready": actor.evidence_ready,
        "fresh_success_holdouts_passed": all(
            holdouts[case_id]["passed"] for case_id in success_ids
        ),
        "fresh_ood_holdouts_rejected": all(
            holdouts[case_id]["passed"] for case_id in rejection_ids
        ),
        "strict_replay_complete": strict_replay,
        "parent_stability_retained": parent_retention,
        "role_backend_evidence_ready": candidate.evidence_ready,
        "role_backend_route_accepted": route.accepted
        and route.selected_backend is RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
        "sim_only_no_torque_authority": True,
    }
    passed = all(gates.values())
    selected_result = cast(dict[str, Any], selected["result"])
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.contextual_finish_intent_portfolio.v1",
        "status": (
            "PASS_CONTEXTUAL_FINISH_INTENT_PORTFOLIO"
            if passed
            else "REJECTED_CONTEXTUAL_FINISH_INTENT_PORTFOLIO"
        ),
        "passed": passed,
        "promotion_eligible": False,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_s204_hash": source204["report_hash"],
        "source_s205_hash": source205["report_hash"],
        "repair_baseline": {
            "result": baseline_result.to_dict(),
            "trajectory": baseline_record,
            "matches_source_failure": baseline_exact,
        },
        "repair": {
            "coarse_candidates": coarse_rows,
            "refinement_seed_candidate_hash": refinement_seed["candidate_hash"],
            "refinement_center": list(refinement_center),
            "refinement_candidates": refinement_rows,
            "selected": selected,
            "selected_replay": {
                "result": replay_result.to_dict(),
                "trajectory": replay_record,
            },
            "selected_exact_replay": selected_exact,
            "discovery_hash": repair_discovery_hash,
            "error_improvement_m": repair_improvement,
        },
        "actor": {
            "file": actor_path.name,
            "file_hash": hash_bytes(actor_path.read_bytes()),
            "actor_hash": actor.actor_hash,
            "algorithm": "nearest_verified_4d_expert_no_interpolation",
            "distinct_context_count": actor.distinct_context_count,
            "distinct_trajectory_count": actor.distinct_trajectory_count,
            "evidence_ready": actor.evidence_ready,
        },
        "holdouts": holdouts,
        "backend_candidate": candidate.to_dict(),
        "backend_candidate_hash": candidate.candidate_hash,
        "route": route.to_dict(),
        "route_hash": route.route_hash,
        "metrics": {
            "repair_candidate_count": len(repair_rows),
            "repair_safe_candidate_count": sum(bool(row["safe"]) for row in repair_rows),
            "repair_precise_candidate_count": sum(bool(row["precise"]) for row in repair_rows),
            "repair_eligible_candidate_count": sum(bool(row["eligible"]) for row in repair_rows),
            "repair_source_target_error_m": repair_source["result"]["target_error_m"],
            "repair_selected_target_error_m": selected_result["target_error_m"],
            "repair_error_improvement_m": repair_improvement,
            "success_holdout_pass_count": sum(
                bool(holdouts[case_id]["passed"]) for case_id in success_ids
            ),
            "success_holdout_count": len(success_ids),
            "rejection_holdout_pass_count": sum(
                bool(holdouts[case_id]["passed"]) for case_id in rejection_ids
            ),
            "rejection_holdout_count": len(rejection_ids),
            "maximum_success_holdout_target_error_m": max(
                float(holdouts[case_id]["result"]["target_error_m"]) for case_id in success_ids
            ),
        },
        "gates": gates,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "partition": "DEVELOPMENT_PLUS_PREREGISTERED_FRESH_HOLDOUT",
            "scope": "LOCAL_CONTEXTUAL_GENERALIZATION_NOT_GLOBAL_FINISHING",
            "plastic_agent_id": "red.finisher",
            "frozen_role_count": 5,
            "joint_torque_owner": ("FROZEN_WHOLE_BODY_KICK_PRIOR_WITH_HASHED_SAFETY_ENVELOPE"),
            "selection_uses_rendered_pixels": False,
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
            "direct_joint_torque_output": False,
            "promotion_authorized": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "contextual-finish-intent-portfolio.json", report)
    return report


def _evaluate_candidates(
    *,
    asset_root: Path,
    output: Path,
    context: dict[str, Any],
    controller: ContextualFinishTargetGrowthConfig,
    physical_target: tuple[float, float, float],
    parent_result: dict[str, Any],
    portfolio_config: ContextualFinishPortfolioConfig,
    criteria: ExtendedFinishIntentRepairConfig,
    candidates: tuple[Any, ...],
    search_stage: str,
    duration: float,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (
            asset_root.expanduser().resolve(),
            _candidate_kwargs(
                context=context,
                controller=controller,
                duration=duration,
                physical_target=physical_target,
                values=candidate.values,
            ),
        )
        for candidate in candidates
    ]
    outcomes = _run_jobs(jobs, workers)
    return [
        _candidate_row(
            output=output,
            search_stage=search_stage,
            candidate=candidate,
            result=result,
            trajectory=trajectory,
            config=criteria,
            parent_result=parent_result,
            portfolio_config=portfolio_config,
        )
        for candidate, (result, trajectory) in zip(candidates, outcomes, strict=True)
    ]


def _sample_features(context: dict[str, Any]) -> tuple[float, ...]:
    return cast(
        tuple[float, ...],
        contextual_finish_intent_features(
            receiver_phase_start_sec=float(context["receiver_phase_start_sec"]),
            prepared_features=cast(tuple[float, ...], tuple(context["features"])),
        ),
    )


def _sample(
    *,
    context: dict[str, Any],
    trajectory_hash: str,
    source_evidence_hash: str,
    control_hash: str,
    physical_target: tuple[float, float, float],
    action_values: tuple[float, float, float, float],
    result: dict[str, Any],
    stability_retained: bool,
    exact_replay: bool,
) -> ContextualFinishIntentSample:
    return ContextualFinishIntentSample(
        context_hash=str(context["context_hash"]),
        trajectory_hash=trajectory_hash,
        control_envelope_hash=control_hash,
        source_evidence_hash=source_evidence_hash,
        features=_sample_features(context),
        action=ContextualFinishIntentAction(*action_values),
        requested_physical_target_m=physical_target,
        observed_crossing_m=(
            physical_target[0],
            float(result["goal_crossing_y_m"]),
            float(result["goal_crossing_z_m"]),
        ),
        target_error_m=float(result["target_error_m"]),
        safe=True,
        stability_retained=stability_retained,
        exact_replay=exact_replay,
    )


def _build_samples(
    *,
    source204: dict[str, Any],
    source205: dict[str, Any],
    request204: dict[str, Any],
    physical_target: tuple[float, float, float],
    control_hash: str,
    repair_discovery_hash: str,
    repair_source: dict[str, Any],
    repair_context: dict[str, Any],
    selected: dict[str, Any],
    selected_exact: bool,
) -> tuple[ContextualFinishIntentSample, ...]:
    values: list[ContextualFinishIntentSample] = []
    for case_id in sorted(source204["success_discovery"]):
        outcome = cast(dict[str, Any], source204["success_discovery"][case_id])
        chosen = cast(dict[str, Any], outcome["selected"])
        result = cast(dict[str, Any], chosen["result"])
        context = cast(dict[str, Any], request204["resolved_contexts"][case_id])
        values.append(
            _sample(
                context=context,
                trajectory_hash=str(chosen["trajectory"]["trajectory_digest"]),
                source_evidence_hash=str(source204["report_hash"]),
                control_hash=control_hash,
                physical_target=physical_target,
                action_values=(
                    float(chosen["policy_target_m"][1]),
                    float(chosen["foot_yaw_offset_rad"]),
                    -0.06,
                    0.01,
                ),
                result=result,
                stability_retained=bool(outcome["stability_retained"]),
                exact_replay=bool(outcome["selected_exact_replay"]),
            )
        )
    source205_selected = cast(dict[str, Any], source205["selected"])
    source205_result = cast(dict[str, Any], source205_selected["result"])
    values.append(
        _sample(
            context=cast(dict[str, Any], request204["resolved_contexts"]["basin-a-sealed"]),
            trajectory_hash=str(source205_selected["trajectory"]["trajectory_digest"]),
            source_evidence_hash=str(source205["report_hash"]),
            control_hash=control_hash,
            physical_target=physical_target,
            action_values=cast(
                tuple[float, float, float, float],
                tuple(source205_selected["action_values"]),
            ),
            result=source205_result,
            stability_retained=bool(source205_selected["stability_retained"]),
            exact_replay=bool(source205["gates"]["independent_exact_replay"]),
        )
    )
    selected_result = cast(dict[str, Any], selected["result"])
    values.append(
        _sample(
            context=repair_context,
            trajectory_hash=str(selected["trajectory"]["trajectory_digest"]),
            source_evidence_hash=repair_discovery_hash,
            control_hash=control_hash,
            physical_target=physical_target,
            action_values=cast(tuple[float, float, float, float], tuple(selected["action_values"])),
            result=selected_result,
            stability_retained=bool(selected["stability_retained"]),
            exact_replay=selected_exact,
        )
    )
    if repair_source["context_hash"] != repair_context["context_hash"]:
        raise ValueError("S206 repair context binding changed")
    return tuple(values)


def _run_holdouts(
    *,
    asset_root: Path,
    output: Path,
    actor: ContextualFinishIntentActor,
    active: ContextualFinishIntentPortfolioConfig,
    controller: ContextualFinishTargetGrowthConfig,
    portfolio_config: ContextualFinishPortfolioConfig,
    resolved: dict[str, Any],
    physical_target: tuple[float, float, float],
    parent_target: tuple[float, float, float],
    parent_yaw: float,
) -> dict[str, Any]:
    success_ids = {context.case_id for context in active.success_holdouts}
    values: dict[str, Any] = {}
    for context in (*active.success_holdouts, *active.rejection_holdouts):
        context_record = cast(dict[str, Any], resolved[context.case_id])
        decision = actor.decide(_sample_features(context_record))
        expects_success = context.case_id in success_ids
        if expects_success and decision.accepted and decision.action is not None:
            action_values = decision.action.values
            action_executed = True
        else:
            action_values = (parent_target[1], parent_yaw, -0.06, 0.01)
            action_executed = False
        kwargs = _candidate_kwargs(
            context=context_record,
            controller=controller,
            duration=active.simulation_duration_sec,
            physical_target=physical_target,
            values=action_values,
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
        parent_kwargs = _candidate_kwargs(
            context=context_record,
            controller=controller,
            duration=active.simulation_duration_sec,
            physical_target=physical_target,
            values=(parent_target[1], parent_yaw, -0.06, 0.01),
        )
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        parent_record = _save_trajectory(
            output / f"holdout-{context.case_id}-parent.npz", parent_trajectory
        )
        result_dict = result.to_dict()
        stability = _stability_retained(result_dict, parent_result.to_dict(), portfolio_config)
        error = result_dict.get("target_error_m")
        pass_error = result_dict.get("pass_delivery_error_m")
        if expects_success:
            passed = bool(
                decision.accepted
                and action_executed
                and _safe_result_dict(result_dict)
                and result.goal_crossed
                and isinstance(error, int | float)
                and not isinstance(error, bool)
                and float(error) <= active.maximum_target_error_m
                and isinstance(pass_error, int | float)
                and not isinstance(pass_error, bool)
                and float(pass_error) <= active.maximum_pass_error_m
                and exact
                and stability
            )
        else:
            passed = bool(
                not decision.accepted
                and decision.route == "CONTEXTUAL_FINISH_INTENT_OOD_FALLBACK"
                and not action_executed
                and _safe_result_dict(result_dict)
                and exact
                and stability
            )
        values[context.case_id] = {
            "context": asdict(context),
            "context_hash": context_record["context_hash"],
            "expected": "PRECISE_EXECUTION" if expects_success else "OOD_REJECTION",
            "decision": asdict(decision),
            "candidate_action_executed": action_executed,
            "executed_action_values": list(action_values),
            "result": result_dict,
            "trajectory": record,
            "replay": {"result": replay_result.to_dict(), "trajectory": replay_record},
            "parent": {"result": parent_result.to_dict(), "trajectory": parent_record},
            "exact_replay": exact,
            "stability_retained": stability,
            "passed": passed,
        }
    return values


def _route_for_candidate(
    candidate: RoleOptionBackendCandidate, finisher_cell_hash: str
) -> RoleOptionBackendRoute:
    if candidate.evidence_ready:
        return RoleOptionBackendRoute(
            agent_id="red.finisher",
            cell_hash=finisher_cell_hash,
            option=PhysicalSoccerOption.SHOOT,
            selected_backend=candidate.backend,
            candidate_hash=candidate.candidate_hash,
            accepted=True,
            reason="ROLE_AND_EVIDENCE_QUALIFIED",
        )
    return RoleOptionBackendRoute(
        agent_id="red.finisher",
        cell_hash=finisher_cell_hash,
        option=PhysicalSoccerOption.SHOOT,
        selected_backend=None,
        candidate_hash=None,
        accepted=False,
        reason="NO_ROLE_QUALIFIED_BACKEND",
    )


def validate_contextual_finish_intent_portfolio(path: Path) -> dict[str, Any]:
    """Reconstruct S206 sources, search, actor decisions and strict replay."""

    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S206 report must be an object")
    claimed_hash = payload.pop("report_hash", None)
    try:
        request_path = report_path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        active = _config_from_dict(cast(dict[str, Any], request["config"]))
        if (
            claimed_hash != hash_json(payload)
            or payload.get("implementation_hash") != _implementation_hash()
            or request.get("implementation_hash") != _implementation_hash()
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or request.get("config_hash") != active.config_hash
            or request.get("coarse_plan_hash") != active.coarse_plan.plan_hash
            or hash_json(request.get("coarse_plan")) != active.coarse_plan.plan_hash
            or request.get("refinement_plan_hash") != active.refinement_plan.plan_hash
            or hash_json(request.get("refinement_plan")) != active.refinement_plan.plan_hash
            or request.get("holdout_partition_committed_before_physics") is not True
            or request.get("feature_scale") != list(DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE)
            or request.get("maximum_support_distance") != 0.35
            or request.get("physics_authority") != "CPU_MUJOCO"
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or request.get("direct_joint_torque_output") is not False
            or request.get("pixels_used_for_scoring") is not False
        ):
            raise ValueError("S206 report integrity changed")
        source204_path = _find_source(
            report_path,
            "s204-contextual-finish-portfolio-*/contextual-finish-portfolio.json",
            str(request["source_s204_file_hash"]),
        )
        source205_path = _find_source(
            report_path,
            "s205-extended-finish-intent-repair-*/extended-finish-intent-repair.json",
            str(request["source_s205_file_hash"]),
        )
        source204 = validate_contextual_finish_portfolio(source204_path)
        source205 = validate_extended_finish_intent_repair(source205_path)
        request204_path = source204_path.parent / "request.json"
        request204 = json.loads(request204_path.read_text(encoding="utf-8"))
        if (
            source204.get("report_hash") != request.get("source_s204_hash")
            or source205.get("report_hash") != request.get("source_s205_hash")
            or payload.get("source_s204_hash") != source204.get("report_hash")
            or payload.get("source_s205_hash") != source205.get("report_hash")
            or source205.get("source_s204_hash") != source204.get("report_hash")
            or hash_bytes(request204_path.read_bytes()) != request.get("source_s204_request_hash")
            or request.get("body_hash") != request204.get("body_hash")
            or request.get("kick_prior_hash") != request204.get("kick_prior_hash")
            or request.get("roster_hash") != request204.get("roster_hash")
            or request.get("finisher_self_model_hash") != request204.get("finisher_self_model_hash")
            or request.get("control_config") != request204.get("control_config")
            or request.get("control_envelope_hash") != request204.get("control_envelope_hash")
            or request.get("physical_target_m") != request204.get("physical_target_m")
            or request.get("parent_policy_target_m") != request204.get("parent_policy_target_m")
            or request.get("parent_foot_yaw_offset_rad")
            != request204.get("parent_foot_yaw_offset_rad")
        ):
            raise ValueError("S206 source lineage changed")
        repair_source = cast(dict[str, Any], source204["holdouts"][active.repair_source_case_id])
        repair_context = cast(
            dict[str, Any],
            request204["resolved_contexts"][active.repair_source_case_id],
        )
        if (
            request.get("repair_source_context") != repair_context
            or request.get("repair_source_context_hash") != repair_source.get("context_hash")
            or request.get("repair_source_trajectory_hash")
            != repair_source["trajectory"]["trajectory_digest"]
            or request.get("repair_source_target_error_m")
            != repair_source["result"]["target_error_m"]
            or request.get("legacy_discovery_stance_offset_y_m") != -0.06
            or request.get("legacy_discovery_foot_pitch_offset_rad") != 0.01
        ):
            raise ValueError("S206 repair source changed")
        baseline = cast(dict[str, Any], payload["repair_baseline"])
        _validate_trajectory(report_path.parent, baseline["trajectory"])
        baseline_exact = bool(
            hash_json(baseline["result"]) == hash_json(repair_source["result"])
            and baseline["trajectory"]["trajectory_digest"]
            == repair_source["trajectory"]["trajectory_digest"]
        )
        if baseline.get("matches_source_failure") is not baseline_exact:
            raise ValueError("S206 repair source replay changed")
        repair = cast(dict[str, Any], payload["repair"])
        coarse_planned = active.coarse_plan.local_candidates(active.repair_warm_start)
        coarse_rows = cast(list[dict[str, Any]], repair["coarse_candidates"])
        parent_result = cast(dict[str, Any], repair_source["parent"]["result"])
        portfolio_config = _portfolio_config_from_dict(cast(dict[str, Any], request204["config"]))
        criteria = ExtendedFinishIntentRepairConfig()
        if len(coarse_rows) != len(coarse_planned):
            raise ValueError("S206 coarse candidate count changed")
        for row, candidate in zip(coarse_rows, coarse_planned, strict=True):
            _validate_candidate_row(
                report_path.parent,
                row,
                candidate,
                "COARSE",
                criteria,
                parent_result,
                portfolio_config,
            )
        refinement_seed = min(coarse_rows, key=_refinement_seed_key)
        refinement_center = cast(tuple[float, ...], tuple(refinement_seed["action_values"]))
        refinement_planned = active.refinement_plan.local_candidates(refinement_center)
        refinement_rows = cast(list[dict[str, Any]], repair["refinement_candidates"])
        if (
            repair.get("refinement_seed_candidate_hash") != refinement_seed["candidate_hash"]
            or repair.get("refinement_center") != list(refinement_center)
            or len(refinement_rows) != len(refinement_planned)
        ):
            raise ValueError("S206 refinement derivation changed")
        for row, candidate in zip(refinement_rows, refinement_planned, strict=True):
            _validate_candidate_row(
                report_path.parent,
                row,
                candidate,
                "REFINEMENT",
                criteria,
                parent_result,
                portfolio_config,
            )
        repair_rows = [*coarse_rows, *refinement_rows]
        selected = min(repair_rows, key=_selection_key)
        selected_replay = cast(dict[str, Any], repair["selected_replay"])
        _validate_trajectory(report_path.parent, selected_replay["trajectory"])
        selected_exact = bool(
            selected["result"] == selected_replay["result"]
            and selected["trajectory"]["trajectory_digest"]
            == selected_replay["trajectory"]["trajectory_digest"]
        )
        improvement = float(repair_source["result"]["target_error_m"]) - float(
            selected["result"]["target_error_m"]
        )
        discovery_body = {
            "source_context_hash": repair_source["context_hash"],
            "coarse_plan_hash": active.coarse_plan.plan_hash,
            "refinement_plan_hash": active.refinement_plan.plan_hash,
            "rows": repair_rows,
            "selected_candidate_hash": selected["candidate_hash"],
            "selected_replay_trajectory_hash": selected_replay["trajectory"]["trajectory_digest"],
            "selected_exact_replay": selected_exact,
        }
        repair_discovery_hash = str(hash_json(discovery_body))
        if (
            repair.get("selected") != selected
            or repair.get("selected_exact_replay") is not selected_exact
            or repair.get("error_improvement_m") != improvement
            or repair.get("discovery_hash") != repair_discovery_hash
        ):
            raise ValueError("S206 repair selection changed")
        actor_record = cast(dict[str, Any], payload["actor"])
        actor_path = report_path.parent / str(actor_record["file"])
        actor = load_contextual_finish_intent_actor(actor_path)
        expected_samples = _build_samples(
            source204=source204,
            source205=source205,
            request204=request204,
            physical_target=cast(tuple[float, float, float], tuple(request["physical_target_m"])),
            control_hash=str(request["control_envelope_hash"]),
            repair_discovery_hash=repair_discovery_hash,
            repair_source=repair_source,
            repair_context=repair_context,
            selected=selected,
            selected_exact=selected_exact,
        )
        if (
            actor.samples != expected_samples
            or actor_record.get("file_hash") != hash_bytes(actor_path.read_bytes())
            or actor_record.get("actor_hash") != actor.actor_hash
            or actor_record.get("algorithm") != "nearest_verified_4d_expert_no_interpolation"
            or actor_record.get("distinct_context_count") != actor.distinct_context_count
            or actor_record.get("distinct_trajectory_count") != actor.distinct_trajectory_count
            or actor_record.get("evidence_ready") is not actor.evidence_ready
        ):
            raise ValueError("S206 actor binding changed")
        resolved = cast(dict[str, Any], request["resolved_holdouts"])
        configured_ids = {
            context.case_id for context in (*active.success_holdouts, *active.rejection_holdouts)
        }
        if set(resolved) != configured_ids:
            raise ValueError("S206 holdout partition changed")
        prior_hashes = {value["context_hash"] for value in request204["resolved_contexts"].values()}
        if prior_hashes.intersection(value["context_hash"] for value in resolved.values()):
            raise ValueError("S206 holdout leaked into prior development")
        holdouts = cast(dict[str, dict[str, Any]], payload["holdouts"])
        success_ids = {context.case_id for context in active.success_holdouts}
        rejection_ids = {context.case_id for context in active.rejection_holdouts}
        for case_id, row in holdouts.items():
            for key in ("trajectory",):
                _validate_trajectory(report_path.parent, row[key])
            _validate_trajectory(report_path.parent, row["replay"]["trajectory"])
            _validate_trajectory(report_path.parent, row["parent"]["trajectory"])
            decision = actor.decide(_sample_features(resolved[case_id]))
            exact = bool(
                row["result"] == row["replay"]["result"]
                and row["trajectory"]["trajectory_digest"]
                == row["replay"]["trajectory"]["trajectory_digest"]
            )
            stability = _stability_retained(
                row["result"], row["parent"]["result"], portfolio_config
            )
            expects_success = case_id in success_ids
            error = row["result"].get("target_error_m")
            pass_error = row["result"].get("pass_delivery_error_m")
            if expects_success:
                passed = bool(
                    decision.accepted
                    and row["candidate_action_executed"] is True
                    and decision.action is not None
                    and row["executed_action_values"] == list(decision.action.values)
                    and _safe_result_dict(row["result"])
                    and row["result"].get("goal_crossed") is True
                    and isinstance(error, int | float)
                    and not isinstance(error, bool)
                    and float(error) <= active.maximum_target_error_m
                    and isinstance(pass_error, int | float)
                    and not isinstance(pass_error, bool)
                    and float(pass_error) <= active.maximum_pass_error_m
                    and exact
                    and stability
                )
            else:
                passed = bool(
                    not decision.accepted
                    and decision.route == "CONTEXTUAL_FINISH_INTENT_OOD_FALLBACK"
                    and row["candidate_action_executed"] is False
                    and row["executed_action_values"]
                    == [
                        request["parent_policy_target_m"][1],
                        request["parent_foot_yaw_offset_rad"],
                        -0.06,
                        0.01,
                    ]
                    and _safe_result_dict(row["result"])
                    and exact
                    and stability
                )
            if (
                row.get("context_hash") != resolved[case_id]["context_hash"]
                or row.get("decision") != asdict(decision)
                or row.get("exact_replay") is not exact
                or row.get("stability_retained") is not stability
                or row.get("passed") is not passed
                or row.get("expected")
                != ("PRECISE_EXECUTION" if expects_success else "OOD_REJECTION")
            ):
                raise ValueError("S206 holdout derivation changed")
        source_hashes = (
            str(source204["report_hash"]),
            str(source205["report_hash"]),
            repair_discovery_hash,
        )
        candidate = RoleOptionBackendCandidate(
            backend=RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
            option=PhysicalSoccerOption.SHOOT,
            artifact_hash=actor.actor_hash,
            evidence_hash=str(hash_json({"sources": source_hashes})),
            distinct_context_count=actor.distinct_context_count,
            distinct_trajectory_count=actor.distinct_trajectory_count,
            strict_replay=baseline_exact
            and selected_exact
            and all(row["exact_replay"] for row in holdouts.values()),
            holdout_passed=all(row["passed"] for row in holdouts.values()),
            parent_retention_passed=selected["stability_retained"]
            and all(row["stability_retained"] for row in holdouts.values()),
        )
        route = _route_for_candidate(candidate, str(request["finisher_cell_hash"]))
        if (
            payload.get("backend_candidate") != candidate.to_dict()
            or payload.get("backend_candidate_hash") != candidate.candidate_hash
            or payload.get("route") != route.to_dict()
            or payload.get("route_hash") != route.route_hash
        ):
            raise ValueError("S206 backend route changed")
        gates = {
            "source_lineage_bound": True,
            "source_failure_exactly_replayed": baseline_exact,
            "second_basin_precise_safe": bool(selected["eligible"]),
            "second_basin_minimum_improvement_met": improvement
            >= active.minimum_repair_improvement_m,
            "second_basin_independent_exact_replay": selected_exact,
            "eight_distinct_physical_experts": actor.distinct_context_count == 8
            and actor.distinct_trajectory_count == 8,
            "actor_evidence_ready": actor.evidence_ready,
            "fresh_success_holdouts_passed": all(
                holdouts[case_id]["passed"] for case_id in success_ids
            ),
            "fresh_ood_holdouts_rejected": all(
                holdouts[case_id]["passed"] for case_id in rejection_ids
            ),
            "strict_replay_complete": candidate.strict_replay,
            "parent_stability_retained": candidate.parent_retention_passed,
            "role_backend_evidence_ready": candidate.evidence_ready,
            "role_backend_route_accepted": route.accepted
            and route.selected_backend is RoleOptionBackend.CONTEXTUAL_FINISH_TARGET,
            "sim_only_no_torque_authority": True,
        }
        metrics = {
            "repair_candidate_count": len(repair_rows),
            "repair_safe_candidate_count": sum(bool(row["safe"]) for row in repair_rows),
            "repair_precise_candidate_count": sum(bool(row["precise"]) for row in repair_rows),
            "repair_eligible_candidate_count": sum(bool(row["eligible"]) for row in repair_rows),
            "repair_source_target_error_m": repair_source["result"]["target_error_m"],
            "repair_selected_target_error_m": selected["result"]["target_error_m"],
            "repair_error_improvement_m": improvement,
            "success_holdout_pass_count": sum(
                bool(holdouts[case_id]["passed"]) for case_id in success_ids
            ),
            "success_holdout_count": len(success_ids),
            "rejection_holdout_pass_count": sum(
                bool(holdouts[case_id]["passed"]) for case_id in rejection_ids
            ),
            "rejection_holdout_count": len(rejection_ids),
            "maximum_success_holdout_target_error_m": max(
                float(holdouts[case_id]["result"]["target_error_m"]) for case_id in success_ids
            ),
        }
        passed = all(gates.values())
        boundary = cast(dict[str, Any], payload.get("evidence_boundary", {}))
        if (
            payload.get("gates") != gates
            or payload.get("metrics") != metrics
            or payload.get("passed") is not passed
            or payload.get("status")
            != (
                "PASS_CONTEXTUAL_FINISH_INTENT_PORTFOLIO"
                if passed
                else "REJECTED_CONTEXTUAL_FINISH_INTENT_PORTFOLIO"
            )
            or payload.get("promotion_eligible") is not False
            or boundary.get("partition") != "DEVELOPMENT_PLUS_PREREGISTERED_FRESH_HOLDOUT"
            or boundary.get("scope") != "LOCAL_CONTEXTUAL_GENERALIZATION_NOT_GLOBAL_FINISHING"
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("selection_uses_rendered_pixels") is not False
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("hardware_command_sent") is not False
            or boundary.get("direct_joint_torque_output") is not False
            or boundary.get("promotion_authorized") is not False
        ):
            raise ValueError("S206 gates or authority changed")
    finally:
        if claimed_hash is not None:
            payload["report_hash"] = claimed_hash
    return payload


def _config_from_dict(value: dict[str, Any]) -> ContextualFinishIntentPortfolioConfig:
    return ContextualFinishIntentPortfolioConfig(
        **{
            **value,
            "repair_policy_target_y_bounds_m": tuple(value["repair_policy_target_y_bounds_m"]),
            "repair_foot_yaw_bounds_rad": tuple(value["repair_foot_yaw_bounds_rad"]),
            "repair_stance_offset_y_bounds_m": tuple(value["repair_stance_offset_y_bounds_m"]),
            "repair_foot_pitch_bounds_rad": tuple(value["repair_foot_pitch_bounds_rad"]),
            "repair_warm_start": tuple(value["repair_warm_start"]),
            "success_holdouts": tuple(
                FinishPortfolioContext(**context) for context in value["success_holdouts"]
            ),
            "rejection_holdouts": tuple(
                FinishPortfolioContext(**context) for context in value["rejection_holdouts"]
            ),
        }
    )


def _find_source(report_path: Path, pattern: str, expected_hash: str) -> Path:
    for candidate in report_path.parents[1].glob(pattern):
        if hash_bytes(candidate.read_bytes()) == expected_hash:
            return candidate
    raise ValueError("S206 source evidence is unavailable")


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
        Path(__file__).parents[1] / "growth" / "contextual_finish_intent.py",
        Path(__file__).parents[1] / "growth" / "bounded_active_search.py",
        Path(__file__).parents[1] / "growth" / "role_option_backend.py",
        Path(__file__).parent / "extended_finish_intent_repair.py",
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
        raise ValueError("S206 evidence output must be new and external")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-s204", type=Path, required=True)
    parser.add_argument("--source-s205", type=Path, required=True)
    parser.add_argument("--source-lead-pass", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_contextual_finish_intent_portfolio(
        asset_root=args.asset_root,
        source_s204_path=args.source_s204,
        source_s205_path=args.source_s205,
        source_lead_pass_dir=args.source_lead_pass,
        source_checkout=args.source_checkout,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(report["status"])
    print(report["report_hash"])


if __name__ == "__main__":
    main()


__all__ = [
    "ContextualFinishIntentPortfolioConfig",
    "run_contextual_finish_intent_portfolio",
    "validate_contextual_finish_intent_portfolio",
]
