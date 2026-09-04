"""Local physical refinement for consumed continuous finisher-plan failures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    RuntimeContactTargetAction,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    RuntimeFinishPlanAction,
    load_runtime_finish_plan_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionGrowthConfig,
    _load_lead_policy,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.prepared_finish_plan_repair import _run_probe


@dataclass(frozen=True)
class ContinuousFinishPlanRepairConfig:
    """Development-only local-search contract; it cannot promote an actor."""

    precision_radius_m: float = 0.10
    expected_failed_contexts: int = 4
    minimum_actions_per_context: int = 8
    target_context_hashes: tuple[str, ...] = ()
    search_strategy: str = "LOCAL_OR_MICRO"
    target_velocity_center_mps: float | None = None
    target_velocity_step_mps: float = 0.0025
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.precision_radius_m)
            or not 0.05 <= self.precision_radius_m <= 0.20
            or not 1 <= self.expected_failed_contexts <= 6
            or not 6 <= self.minimum_actions_per_context <= 16
            or len(set(self.target_context_hashes)) != len(self.target_context_hashes)
            or any(
                not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
                for value in self.target_context_hashes
            )
            or self.search_strategy
            not in {
                "LOCAL_OR_MICRO",
                "STANCE_COVERAGE",
                "STANCE_EDGE_REFINE",
                "CONTACT_EDGE_REFINE",
                "TARGET_VELOCITY_COVERAGE",
                "TARGET_VELOCITY_EDGE_REFINE",
                "TARGET_VELOCITY_BRACKET_REFINE",
            }
            or (
                self.search_strategy == "TARGET_VELOCITY_BRACKET_REFINE"
                and (
                    self.target_velocity_center_mps is None
                    or not math.isfinite(self.target_velocity_center_mps)
                    or not 0.02 <= self.target_velocity_center_mps <= 5.98
                )
            )
            or (
                self.search_strategy != "TARGET_VELOCITY_BRACKET_REFINE"
                and self.target_velocity_center_mps is not None
            )
            or not math.isfinite(self.target_velocity_step_mps)
            or not 0.00025 <= self.target_velocity_step_mps <= 0.05
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("continuous finish plan repair config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_continuous_finish_plan_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    consumed_exam_path: Path,
    finish_plan_actor_path: Path,
    finish_plan_training_report_path: Path,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    config: ContinuousFinishPlanRepairConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
    seed_repair_path: Path | None = None,
) -> dict[str, Any]:
    """Probe bounded neighborhoods around each failed parent-anchored decision."""

    if not 1 <= workers <= 4:
        raise ValueError("continuous finish plan repair workers must be in [1, 4]")
    active = config or ContinuousFinishPlanRepairConfig()
    exam_path = consumed_exam_path.expanduser().resolve()
    exam = _bound_json(exam_path)
    request_path = exam_path.parent / "request.json"
    rows = exam.get("rows")
    if (
        exam.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam.v1"
        or exam.get("promotion_eligible") is not False
        or not request_path.is_file()
        or hash_bytes(request_path.read_bytes()) != exam.get("request_hash")
        or not isinstance(rows, list)
        or len(rows) != 6
        or not all(row.get("exact_replay") is True for row in rows if isinstance(row, dict))
    ):
        raise ValueError("continuous finish plan repair source exam is invalid")
    failed_rows = [
        row
        for row in cast(list[dict[str, Any]], rows)
        if not row["candidate"]["quality"]["strict_chain_passed"]
        and (
            not active.target_context_hashes
            or row.get("context_hash") in active.target_context_hashes
        )
    ]
    if active.target_context_hashes and active.target_context_hashes != tuple(
        row.get("context_hash") for row in failed_rows
    ):
        raise ValueError("continuous finish plan repair target contexts changed")
    if len(failed_rows) != active.expected_failed_contexts:
        raise ValueError("continuous finish plan repair failure count changed")
    seed_repair: dict[str, Any] | None = None
    seed_repair_file_hash: str | None = None
    if seed_repair_path is not None:
        resolved_seed = seed_repair_path.expanduser().resolve()
        seed_repair = _bound_json(resolved_seed)
        seed_rows = seed_repair.get("rows")
        if (
            seed_repair.get("schema_version") != "rosclaw_soccer.continuous_finish_plan_repair.v1"
            or seed_repair.get("promotion_eligible") is not False
            or seed_repair.get("source_exam_hash") != exam.get("report_hash")
            or not isinstance(seed_rows, list)
        ):
            raise ValueError("continuous finish plan seed repair is invalid")
        seed_repair_file_hash = hash_bytes(resolved_seed.read_bytes())

    actor_path = finish_plan_actor_path.expanduser().resolve()
    actor = load_runtime_finish_plan_actor(actor_path)
    training_path = finish_plan_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    lead, lead_source = _load_lead_policy(source_s95_dir)
    if (
        actor.continuous_policy is None
        or training.get("status") != "PASS_CONTINUOUS_RUNTIME_FINISH_PLAN_CALIBRATION"
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or training.get("parent_actor_hash") != actor.continuous_policy.parent_actor_hash
        or (
            seed_repair is not None
            and seed_repair.get("finish_plan_actor_hash") != actor.actor_hash
        )
        or actor.neural_contact_actor_hash != neural.actor_hash
        or actor.contact_handoff_actor_hash != handoff.actor_hash
        or actor.body_hash != qualification.body_hash
        or actor.kick_prior_hash != qualification.kick_prior_hash
        or actor.roster_hash != base.roster_hash
        or actor.finisher_self_model_hash != base.finisher_self_model_hash
        or base.body_hash != qualification.body_hash
        or neural.body_hash != qualification.body_hash
    ):
        raise ValueError("continuous finish plan repair lineage changed")

    jobs_spec: list[tuple[dict[str, Any], RuntimeFinishPlanAction]] = []
    context_actions: dict[str, tuple[RuntimeFinishPlanAction, ...]] = {}
    for row in failed_rows:
        features = _features_from_row(lead, row)
        decision = actor.decide(features)
        if not decision.accepted or decision.action is None:
            raise ValueError("continuous finish plan repair parent decision is unavailable")
        seeds = (
            (decision.action,)
            if seed_repair is None
            else _seed_actions(
                cast(list[dict[str, Any]], seed_repair["rows"]),
                str(row["context_hash"]),
            )
        )
        proposed_actions = (
            _stance_coverage_actions(seeds)
            if active.search_strategy == "STANCE_COVERAGE"
            else _stance_edge_refinement_actions(seeds)
            if active.search_strategy == "STANCE_EDGE_REFINE"
            else _contact_edge_refinement_actions(seeds)
            if active.search_strategy == "CONTACT_EDGE_REFINE"
            else _target_velocity_coverage_actions(seeds)
            if active.search_strategy == "TARGET_VELOCITY_COVERAGE"
            else _target_velocity_edge_refinement_actions(seeds)
            if active.search_strategy == "TARGET_VELOCITY_EDGE_REFINE"
            else _target_velocity_bracket_actions(
                seeds,
                center=active.target_velocity_center_mps,
                step=active.target_velocity_step_mps,
            )
            if active.search_strategy == "TARGET_VELOCITY_BRACKET_REFINE"
            else (
                _local_refinement_actions(decision.action)
                if seed_repair is None
                else _micro_refinement_actions(seeds)
            )
        )
        actions = tuple(
            action
            for action in proposed_actions
            if neural.target_supported(action.target.target_foot_velocity_xyz_mps)
        )
        if len(actions) < active.minimum_actions_per_context:
            raise ValueError("continuous finish plan repair action bank is incomplete")
        context_actions[str(row["context_hash"])] = actions
        jobs_spec.extend((row, action) for action in actions)

    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_finish_plan_repair_request.v1",
        "partition": "CONSUMED_LOCAL_CONTINUOUS_REFINEMENT",
        "source_exam_hash": exam["report_hash"],
        "source_exam_file_hash": hash_bytes(exam_path.read_bytes()),
        "seed_repair_hash": None if seed_repair is None else seed_repair["report_hash"],
        "seed_repair_file_hash": seed_repair_file_hash,
        "failed_context_hashes": [row["context_hash"] for row in failed_rows],
        "context_actions": {
            context_hash: [asdict(action) for action in actions]
            for context_hash, actions in context_actions.items()
        },
        "finish_plan_actor_hash": actor.actor_hash,
        "finish_plan_actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "finish_plan_training_report_hash": training["report_hash"],
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            base_path,
            neural_path,
            output,
            index,
            row,
            action,
            handoff.selected_offset_frames,
            quality,
        )
        for index, (row, action) in enumerate(jobs_spec)
    )
    if workers == 1:
        probe_rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            probe_rows = list(executor.map(_run_probe, jobs))
    selected = [_select(probe_rows, str(source["context_hash"]), active) for source in failed_rows]
    recovered = {row["context_hash"] for row in probe_rows if row["quality"]["strict_chain_passed"]}
    precise = [row for row in probe_rows if _precision_passed(row, active)]
    gates = {
        "all_consumed_failures_recovered": len(recovered) == len(failed_rows),
        "selected_safe_all": all(row["quality"]["safe"] for row in selected),
        "selected_strict_all": all(row["quality"]["strict_chain_passed"] for row in selected),
        "at_least_two_precise_goals": len(precise) >= 2,
        "neural_actor_realized_all": all(row["neural_actor_active"] for row in probe_rows),
        "teacher_and_scripted_contact_absent": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in probe_rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in probe_rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_finish_plan_repair.v1",
        "status": (
            "PASS_CONTINUOUS_FINISH_PLAN_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_CONTINUOUS_FINISH_PLAN_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "partition": request["partition"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_exam_hash": exam["report_hash"],
        "finish_plan_actor_hash": actor.actor_hash,
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": actor.roster_hash,
        "finisher_self_model_hash": actor.finisher_self_model_hash,
        "metrics": {
            "probe_count": len(probe_rows),
            "context_count": len(failed_rows),
            "safe_count": sum(bool(row["quality"]["safe"]) for row in probe_rows),
            "strict_success_count": sum(
                bool(row["quality"]["strict_chain_passed"]) for row in probe_rows
            ),
            "recovered_context_count": len(recovered),
            "precise_goal_count": len(precise),
            "best_goal_target_error_m": min(
                (float(row["result"]["target_error_m"]) for row in precise),
                default=None,
            ),
        },
        "selected": selected,
        "gates": gates,
        "rows": probe_rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "repair-report.json", report)
    return report


def _local_refinement_actions(
    parent: RuntimeFinishPlanAction,
) -> tuple[RuntimeFinishPlanAction, ...]:
    receive = parent.receive
    target = parent.target.target_foot_velocity_xyz_mps
    deltas = (
        (-0.020, 0.000, 0, 0.000, 0.0),
        (-0.010, 0.000, 0, 0.000, 0.0),
        (0.010, 0.000, 0, 0.000, 0.0),
        (0.000, -0.020, 0, 0.000, 0.0),
        (0.000, -0.010, 0, 0.000, 0.0),
        (0.000, 0.010, 0, 0.000, 0.0),
        (0.000, 0.000, -2, 0.000, 0.0),
        (0.000, 0.000, 2, 0.000, 0.0),
        (0.000, 0.000, 0, -0.010, 0.0),
        (0.000, 0.000, 0, 0.010, 0.0),
        (0.000, 0.000, 0, 0.000, -0.5),
        (0.000, 0.000, 0, 0.000, 0.5),
        (-0.010, -0.010, -1, -0.010, 0.5),
        (-0.010, 0.010, 1, 0.010, -0.5),
    )
    actions: dict[str, RuntimeFinishPlanAction] = {}
    for stance_x, stance_y, frame, yaw, target_y in deltas:
        candidate = RuntimeFinishPlanAction(
            receive=replace(
                receive,
                stance_offset_x_m=float(
                    max(-0.12, min(0.12, receive.stance_offset_x_m + stance_x))
                ),
                stance_offset_y_m=float(
                    max(-0.12, min(0.12, receive.stance_offset_y_m + stance_y))
                ),
                contact_policy_frame=max(238, min(258, receive.contact_policy_frame + frame)),
                foot_yaw_offset_rad=float(max(-0.12, min(0.12, receive.foot_yaw_offset_rad + yaw))),
            ),
            target=RuntimeContactTargetAction(
                (
                    target[0],
                    float(max(-6.0, min(6.0, target[1] + target_y))),
                    target[2],
                )
            ),
        )
        if candidate != parent:
            actions[candidate.action_hash] = candidate
    return tuple(actions[key] for key in sorted(actions))


def _seed_actions(
    rows: list[dict[str, Any]], context_hash: str
) -> tuple[RuntimeFinishPlanAction, ...]:
    candidates = [row for row in rows if row.get("context_hash") == context_hash]
    if not candidates:
        raise ValueError("continuous finish plan seed context is missing")
    strict = min(
        candidates,
        key=lambda row: (
            not row["quality"]["strict_chain_passed"],
            _target_error(row),
            row["action_hash"],
        ),
    )
    precision = min(
        candidates,
        key=lambda row: (
            not row["quality"]["clear_outcome"],
            _target_error(row),
            not row["quality"]["intended_foot_contact"],
            row["action_hash"],
        ),
    )
    actions: list[RuntimeFinishPlanAction] = []
    for row in (strict, precision):
        payload = cast(dict[str, Any], row["action"])
        receive = RuntimeReceiveAction(**cast(dict[str, Any], payload["receive"]))
        target_payload = dict(cast(dict[str, Any], payload["target"]))
        target_payload["target_foot_velocity_xyz_mps"] = tuple(
            target_payload["target_foot_velocity_xyz_mps"]
        )
        action = RuntimeFinishPlanAction(
            receive=receive,
            target=RuntimeContactTargetAction(**target_payload),
            activation_ceiling=str(payload.get("activation_ceiling", "SIM_ONLY")),
            direct_joint_torque_output=bool(payload.get("direct_joint_torque_output", False)),
        )
        if action not in actions:
            actions.append(action)
    return tuple(actions)


def _micro_refinement_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    deltas = (
        # arrival frames, tolerance, stance x/y, policy frame, yaw, pitch,
        # target x/y. Timing comes first because adjacent contexts showed a
        # four-frame arrival bifurcation despite nearly identical geometry.
        (6, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (12, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (-6, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (6, 0.02, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (12, 0.02, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (0, 0.02, 0.000, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (0, 0.00, 0.000, 0.000, -3, 0.000, 0.000, 0.00, 0.00),
        (0, 0.00, 0.000, 0.000, 3, 0.000, 0.000, 0.00, 0.00),
        (6, 0.00, 0.000, 0.005, 0, 0.000, 0.000, 0.00, 0.00),
        (6, 0.00, -0.010, 0.000, 0, 0.000, 0.000, 0.00, 0.00),
        (6, 0.00, 0.000, 0.000, 0, -0.006, 0.000, 0.00, 0.00),
        (6, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.00, -0.50),
        (12, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.00, -0.50),
        (6, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.50, 0.00),
        (12, 0.00, 0.000, 0.000, 0, 0.000, 0.000, 0.50, 0.00),
        (6, 0.00, 0.000, 0.000, 0, 0.000, -0.005, 0.00, 0.00),
    )
    actions: dict[str, RuntimeFinishPlanAction] = {}
    # Interleave seeds so the 16-action budget cannot silently collapse onto
    # whichever strict/precision seed happened to be listed first.
    allowed_advances = (0, 6, 12, 18, 24, 30)
    for advance, tolerance, stance_x, stance_y, frame, yaw, pitch, target_x, target_y in deltas:
        for seed in seeds:
            receive = seed.receive
            target = seed.target.target_foot_velocity_xyz_mps
            desired_advance = receive.maximum_arrival_advance_frames + advance
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    receive,
                    maximum_arrival_advance_frames=min(
                        allowed_advances,
                        key=lambda value: abs(value - desired_advance),
                    ),
                    arrival_alignment_tolerance_sec=float(
                        max(
                            0.02,
                            min(0.12, receive.arrival_alignment_tolerance_sec + tolerance),
                        )
                    ),
                    stance_offset_x_m=float(
                        max(-0.12, min(0.12, receive.stance_offset_x_m + stance_x))
                    ),
                    stance_offset_y_m=float(
                        max(-0.12, min(0.12, receive.stance_offset_y_m + stance_y))
                    ),
                    contact_policy_frame=max(238, min(258, receive.contact_policy_frame + frame)),
                    foot_yaw_offset_rad=float(
                        max(-0.12, min(0.12, receive.foot_yaw_offset_rad + yaw))
                    ),
                    foot_pitch_offset_rad=float(
                        max(-0.08, min(0.08, receive.foot_pitch_offset_rad + pitch))
                    ),
                ),
                target=RuntimeContactTargetAction(
                    (
                        float(max(5.0, min(12.0, target[0] + target_x))),
                        float(max(-6.0, min(6.0, target[1] + target_y))),
                        target[2],
                    )
                ),
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _stance_coverage_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Cover the stance envelope when millimetre-local search is non-causal."""

    stance_values = (-0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12)
    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        for stance_y in stance_values:
            candidate = RuntimeFinishPlanAction(
                receive=replace(seed.receive, stance_offset_y_m=stance_y),
                target=seed.target,
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
        for stance_x in (-0.12, -0.08, 0.0, 0.04, 0.08, 0.12):
            candidate = RuntimeFinishPlanAction(
                receive=replace(seed.receive, stance_offset_x_m=stance_x),
                target=seed.target,
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
        candidate = RuntimeFinishPlanAction(
            receive=replace(seed.receive, contact_policy_frame=258),
            target=seed.target,
        )
        if candidate not in seeds:
            actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _stance_edge_refinement_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Resolve a narrow contact bifurcation around a safe stance seed."""

    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        for delta_y in (-0.010, -0.005, 0.005, 0.010, 0.015, 0.020, 0.025):
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    seed.receive,
                    stance_offset_y_m=float(
                        max(-0.12, min(0.12, seed.receive.stance_offset_y_m + delta_y))
                    ),
                ),
                target=seed.target,
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
        for delta_x in (-0.020, -0.010):
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    seed.receive, stance_offset_x_m=seed.receive.stance_offset_x_m + delta_x
                ),
                target=seed.target,
            )
            actions.setdefault(candidate.action_hash, candidate)
        for frame in (253, 254, 256, 257, 258):
            candidate = RuntimeFinishPlanAction(
                receive=replace(seed.receive, contact_policy_frame=frame),
                target=seed.target,
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
        for yaw in (-0.12, -0.11):
            candidate = RuntimeFinishPlanAction(
                receive=replace(seed.receive, foot_yaw_offset_rad=yaw),
                target=seed.target,
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _contact_edge_refinement_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Cross a measured post/ball contact boundary without broad exploration."""

    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        for delta_y in (-0.003, -0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002, 0.003):
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    seed.receive,
                    stance_offset_y_m=float(
                        max(-0.12, min(0.12, seed.receive.stance_offset_y_m + delta_y))
                    ),
                ),
                target=seed.target,
            )
            actions.setdefault(candidate.action_hash, candidate)
        for delta_yaw in (-0.003, -0.002, -0.001, 0.001, 0.002, 0.003):
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    seed.receive,
                    foot_yaw_offset_rad=float(
                        max(-0.12, min(0.12, seed.receive.foot_yaw_offset_rad + delta_yaw))
                    ),
                ),
                target=seed.target,
            )
            actions.setdefault(candidate.action_hash, candidate)
        for delta_pitch in (-0.002, 0.002):
            candidate = RuntimeFinishPlanAction(
                receive=replace(
                    seed.receive,
                    foot_pitch_offset_rad=float(
                        max(
                            -0.08,
                            min(0.08, seed.receive.foot_pitch_offset_rad + delta_pitch),
                        )
                    ),
                ),
                target=seed.target,
            )
            actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _target_velocity_coverage_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Sweep the learned lateral target support around a proven contact pose."""

    lateral_targets = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        4.50,
        5.00,
        6.00,
    )
    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        target = seed.target.target_foot_velocity_xyz_mps
        for lateral in lateral_targets:
            candidate = RuntimeFinishPlanAction(
                receive=seed.receive,
                target=RuntimeContactTargetAction((target[0], lateral, target[2])),
            )
            if candidate not in seeds:
                actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _target_velocity_edge_refinement_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Resolve a narrow target-velocity interval around a strict seed."""

    deltas = (
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
    )
    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        target = seed.target.target_foot_velocity_xyz_mps
        for delta in deltas:
            candidate = RuntimeFinishPlanAction(
                receive=seed.receive,
                target=RuntimeContactTargetAction(
                    (target[0], float(max(-6.0, min(6.0, target[1] + delta))), target[2])
                ),
            )
            actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _target_velocity_bracket_actions(
    seeds: tuple[RuntimeFinishPlanAction, ...],
    *,
    center: float | None,
    step: float,
) -> tuple[RuntimeFinishPlanAction, ...]:
    """Densely resolve a measured target-velocity response bracket."""

    if center is None or not math.isfinite(center) or not math.isfinite(step) or step <= 0.0:
        raise ValueError("target velocity bracket center is missing")
    offsets = tuple(step * (index - 7) for index in range(16))
    actions: dict[str, RuntimeFinishPlanAction] = {}
    for seed in seeds:
        target = seed.target.target_foot_velocity_xyz_mps
        for offset in offsets:
            candidate = RuntimeFinishPlanAction(
                receive=seed.receive,
                target=RuntimeContactTargetAction(
                    (target[0], float(max(-6.0, min(6.0, center + offset))), target[2])
                ),
            )
            actions.setdefault(candidate.action_hash, candidate)
    return tuple(actions.values())[:16]


def _features_from_row(lead: Any, row: dict[str, Any]) -> tuple[float, ...]:
    from rosclaw_soccer.training.continuous_finish_plan_growth import (
        _prepared_features_from_row,
    )

    return _prepared_features_from_row(lead, row)


def _precision_passed(row: dict[str, Any], config: ContinuousFinishPlanRepairConfig) -> bool:
    error = row["result"].get("target_error_m")
    return bool(
        row["quality"]["strict_chain_passed"]
        and row["result"]["goal_crossed"]
        and isinstance(error, int | float)
        and not isinstance(error, bool)
        and math.isfinite(float(error))
        and float(error) <= config.precision_radius_m
    )


def _select(
    rows: list[dict[str, Any]],
    context_hash: str,
    config: ContinuousFinishPlanRepairConfig,
) -> dict[str, Any]:
    return min(
        (row for row in rows if row["context_hash"] == context_hash),
        key=lambda row: (
            not row["quality"]["strict_chain_passed"],
            not row["quality"]["safe"],
            not _precision_passed(row, config),
            not row["quality"]["intended_foot_contact"],
            not row["quality"]["clear_outcome"],
            _target_error(row),
            float(row["result"]["shooter_post_contact_support_foot_slip_m"]),
            row["action_hash"],
        ),
    )


def _target_error(row: dict[str, Any]) -> float:
    value = row["result"].get("target_error_m")
    return (
        float(value)
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else 1.0e6
    )


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parent / "prepared_finish_plan_repair.py",
        Path(__file__).parents[1] / "growth" / "runtime_finish_plan_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("continuous finish plan repair output must be new and external")
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
    "ContinuousFinishPlanRepairConfig",
    "run_continuous_finish_plan_repair",
]
