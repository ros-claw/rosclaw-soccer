"""Strict local holdout exam for incremental finisher-plan precision growth."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.dynamic_lead_pass import DynamicLeadPassPolicy
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    G1RuntimeFinishPlanActor,
    load_runtime_finish_plan_actor,
    prepared_finish_plan_features,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction
from rosclaw_soccer.training.runtime_finish_plan_exam import (
    run_runtime_finish_plan_exam,
    validate_runtime_finish_plan_exam,
)


@dataclass(frozen=True)
class RuntimeFinishPlanPrecisionExamConfig:
    """Frozen S170 promotion thresholds."""

    precision_radius_m: float = 0.10
    minimum_strict_successes: int = 5
    minimum_precise_goals: int = 3
    minimum_strict_gain: int = 2
    minimum_normalized_training_distance: float = 0.02
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.precision_radius_m)
            or not 0.05 <= self.precision_radius_m <= 0.10
            or not 5 <= self.minimum_strict_successes <= 6
            or not 3 <= self.minimum_precise_goals <= 5
            or not 2 <= self.minimum_strict_gain <= 5
            or not math.isfinite(self.minimum_normalized_training_distance)
            or not 0.02 <= self.minimum_normalized_training_distance <= 0.25
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("runtime finish plan precision exam config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def fresh_runtime_finish_plan_precision_holdouts() -> tuple[
    tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...
]:
    """Corrected local perturbations reserved for the S170-v2 sealed exam."""

    origin = (5.10, -0.16406006503921598, 0.0)
    values = (
        ("s170.precision.v2.00", 1.20933, -0.16533, 0.10083, 0.04),
        ("s170.precision.v2.01", 1.20943, -0.16543, 0.10093, 0.02),
        ("s170.precision.v2.02", 1.20960, -0.16560, 0.10110, 0.04),
        ("s170.precision.v2.03", 1.20990, -0.16590, 0.10140, 0.04),
        ("s170.precision.v2.04", 1.21345, -0.16945, 0.10370, 0.06),
        # v1 used (1.21380, -0.16980, 0.10390), whose runtime feature
        # vector exactly duplicated a training memory despite a new context
        # hash.  This replacement stays local but clears the declared
        # normalized-distance floor before any v2 outcome is observed.
        ("s170.precision.v2.05", 1.21390, -0.16990, 0.10398, 0.06),
    )
    return tuple(
        (
            CausalTransitionContext(
                case_id,
                origin,
                -0.0875,
                1.3005,
                (ball_x, ball_y),
                0.80,
                friction,
            ),
            PlaymakerPassProbeAction(body_yaw_correction_rad=yaw),
        )
        for case_id, ball_x, ball_y, friction, yaw in values
    )


def run_runtime_finish_plan_precision_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    finish_plan_actor_path: Path,
    finish_plan_training_report_path: Path,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    config: RuntimeFinishPlanPrecisionExamConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run a matched exam, then apply the stronger 5/6 and 0.10 m gates."""

    active = config or RuntimeFinishPlanPrecisionExamConfig()
    cases = fresh_runtime_finish_plan_precision_holdouts()
    actor_path = finish_plan_actor_path.expanduser().resolve()
    actor = load_runtime_finish_plan_actor(actor_path)
    training_path = finish_plan_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    source_root = source_s95_dir.expanduser().resolve()
    lead, lead_evidence = _load_lead_policy(source_root)
    feature_vectors = _holdout_feature_vectors(cases, lead)
    feature_partition = _feature_partition(actor, feature_vectors)
    context_hashes = [context.context_hash for context, _ in cases]
    training_context_hashes = {
        memory.context_hash for memory in (*actor.successful_memories, *actor.failed_memories)
    }
    if (
        training.get("status") != "PASS_RUNTIME_FINISH_PLAN_TRAINING"
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or training_context_hashes.intersection(context_hashes)
        or feature_partition["minimum_normalized_training_distance"]
        < active.minimum_normalized_training_distance
    ):
        raise ValueError("runtime finish plan precision partition is not fresh")

    output = _new_external_output(output_dir)
    actor_snapshot = output / "runtime-finish-plan-actor.json"
    training_snapshot = output / "training-report.json"
    actor_snapshot.write_bytes(actor_path.read_bytes())
    training_snapshot.write_bytes(training_path.read_bytes())
    lead_snapshot = output / "lead-source"
    lead_snapshot.mkdir()
    for name in ("evidence.json", "dynamic-lead-pass-policy.json"):
        (lead_snapshot / name).write_bytes((source_root / name).read_bytes())
    snapshot_lead, snapshot_evidence = _load_lead_policy(lead_snapshot)
    if (
        snapshot_lead.artifact_hash != lead.artifact_hash
        or snapshot_evidence["evidence_hash"] != lead_evidence["evidence_hash"]
    ):
        raise ValueError("runtime finish plan precision lead source snapshot changed")
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_precision_exam_request.v2",
        "partition": "SEALED_LOCAL_PRECISION_HOLDOUT_V2",
        "sealed": True,
        "cases": [
            {"context": asdict(context), "playmaker_action": asdict(playmaker)}
            for context, playmaker in cases
        ],
        "context_hashes": context_hashes,
        "training_context_hashes_commitment": hash_json(sorted(training_context_hashes)),
        "training_context_count": len(training_context_hashes),
        **feature_partition,
        "finish_plan_actor_hash": actor.actor_hash,
        "finish_plan_actor_file": actor_snapshot.name,
        "finish_plan_actor_file_hash": hash_bytes(actor_snapshot.read_bytes()),
        "finish_plan_training_report_file": training_snapshot.name,
        "finish_plan_training_report_hash": training["report_hash"],
        "finish_plan_training_report_file_hash": hash_bytes(training_snapshot.read_bytes()),
        "source_s95_evidence_hash": lead_evidence["evidence_hash"],
        "source_s95_evidence_file": "lead-source/evidence.json",
        "source_s95_evidence_file_hash": hash_bytes((lead_snapshot / "evidence.json").read_bytes()),
        "source_s95_policy_hash": lead.artifact_hash,
        "source_s95_policy_file": "lead-source/dynamic-lead-pass-policy.json",
        "source_s95_policy_file_hash": hash_bytes(
            (lead_snapshot / "dynamic-lead-pass-policy.json").read_bytes()
        ),
        "config": asdict(active),
        "config_hash": active.config_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    matched = run_runtime_finish_plan_exam(
        asset_root=asset_root,
        source_s95_dir=source_s95_dir,
        finish_plan_actor_path=actor_snapshot,
        finish_plan_training_report_path=training_snapshot,
        base_target_actor_path=base_target_actor_path,
        neural_actor_path=neural_actor_path,
        handoff_actor_path=handoff_actor_path,
        output_dir=output / "matched-exam",
        cases=cases,
        quality_config=quality_config,
        sealed=True,
        workers=workers,
    )
    metrics, gates = _derive_metrics_and_gates(matched, active)
    passed = all(gates.values())
    matched_path = output / "matched-exam" / "exam-report.json"
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.runtime_finish_plan_precision_exam.v2",
        "status": (
            "PASS_RUNTIME_FINISH_PLAN_PRECISION_FRESH_HOLDOUT"
            if passed
            else "REJECTED_RUNTIME_FINISH_PLAN_PRECISION"
        ),
        "sealed": True,
        "promotion_eligible": passed,
        "partition": request["partition"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "finish_plan_actor_hash": actor.actor_hash,
        "matched_exam_report_file": str(matched_path.relative_to(output)),
        "matched_exam_report_file_hash": hash_bytes(matched_path.read_bytes()),
        "matched_exam_report_hash": matched["report_hash"],
        "metrics": metrics,
        "gates": gates,
        "implementation_hash": _implementation_hash(),
        "evidence_boundary": {
            "whole_body_g1_count": 3,
            "local_perturbation_scope_only": True,
            "runtime_feature_partition_fresh": True,
            "minimum_normalized_training_distance": feature_partition[
                "minimum_normalized_training_distance"
            ],
            "one_shared_solver_and_ball": True,
            "joint_torque_owned_only_by_frozen_neural_actor": True,
            "pixels_used_for_scoring": False,
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "precision-exam-report.json", report)
    return report


def validate_runtime_finish_plan_precision_exam(path: Path) -> dict[str, Any]:
    """Recompute every strong gate from a bound matched-exam artifact."""

    source = path.expanduser().resolve()
    report = _bound_json(source)
    request_path = source.parent / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("runtime finish plan precision request must be an object")
    config_payload = request.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("runtime finish plan precision config is missing")
    config = RuntimeFinishPlanPrecisionExamConfig(**config_payload)
    actor_path = _local_artifact(source.parent, request.get("finish_plan_actor_file"))
    training_path = _local_artifact(source.parent, request.get("finish_plan_training_report_file"))
    actor = load_runtime_finish_plan_actor(actor_path)
    training = _bound_json(training_path)
    lead_evidence_path = _local_artifact(source.parent, request.get("source_s95_evidence_file"))
    lead_policy_path = _local_artifact(source.parent, request.get("source_s95_policy_file"))
    if lead_evidence_path.parent != lead_policy_path.parent:
        raise ValueError("runtime finish plan precision lead source snapshot is split")
    lead, lead_evidence = _load_lead_policy(lead_evidence_path.parent)
    matched_path = _local_artifact(source.parent, report.get("matched_exam_report_file"))
    matched = validate_runtime_finish_plan_exam(matched_path)
    matched_request = json.loads((matched_path.parent / "request.json").read_text(encoding="utf-8"))
    training_context_hashes = {
        memory.context_hash for memory in (*actor.successful_memories, *actor.failed_memories)
    }
    context_hashes = request.get("context_hashes")
    cases_payload = request.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError("runtime finish plan precision cases are missing")
    cases = _decode_cases(cases_payload)
    expected_cases = fresh_runtime_finish_plan_precision_holdouts()
    feature_partition = _feature_partition(actor, _holdout_feature_vectors(cases, lead))
    metrics, gates = _derive_metrics_and_gates(matched, config)
    passed = all(gates.values())
    expected_status = (
        "PASS_RUNTIME_FINISH_PLAN_PRECISION_FRESH_HOLDOUT"
        if passed
        else "REJECTED_RUNTIME_FINISH_PLAN_PRECISION"
    )
    boundary = report.get("evidence_boundary")
    if (
        request.get("schema_version")
        != "rosclaw_soccer.runtime_finish_plan_precision_exam_request.v2"
        or request.get("partition") != "SEALED_LOCAL_PRECISION_HOLDOUT_V2"
        or request.get("sealed") is not True
        or cases != expected_cases
        or not isinstance(context_hashes, list)
        or len(context_hashes) != 6
        or len(set(context_hashes)) != 6
        or training_context_hashes.intersection(context_hashes)
        or request.get("training_context_hashes_commitment")
        != hash_json(sorted(training_context_hashes))
        or request.get("training_context_count") != len(training_context_hashes)
        or any(request.get(key) != value for key, value in feature_partition.items())
        or feature_partition["minimum_normalized_training_distance"]
        < config.minimum_normalized_training_distance
        or request.get("finish_plan_actor_hash") != actor.actor_hash
        or request.get("finish_plan_actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or request.get("finish_plan_training_report_hash") != training.get("report_hash")
        or request.get("finish_plan_training_report_file_hash")
        != hash_bytes(training_path.read_bytes())
        or training.get("actor_hash") != actor.actor_hash
        or training.get("status") != "PASS_RUNTIME_FINISH_PLAN_TRAINING"
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or request.get("source_s95_evidence_hash") != lead_evidence.get("evidence_hash")
        or request.get("source_s95_evidence_file_hash")
        != hash_bytes(lead_evidence_path.read_bytes())
        or request.get("source_s95_policy_hash") != lead.artifact_hash
        or request.get("source_s95_policy_file_hash") != hash_bytes(lead_policy_path.read_bytes())
        or request.get("config_hash") != config.config_hash
        or request.get("implementation_hash") != _implementation_hash()
        or request.get("physics_authority") != "CPU_MUJOCO"
        or request.get("activation_ceiling") != "SIM_ONLY"
        or request.get("hardware_command_sent") is not False
        or matched_request.get("context_hashes") != context_hashes
        or matched_request.get("cases") != request.get("cases")
        or matched_request.get("finish_plan_actor_hash") != actor.actor_hash
        or matched_request.get("finish_plan_actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or matched_request.get("source_s95_evidence_hash") != lead_evidence.get("evidence_hash")
        or report.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_precision_exam.v2"
        or report.get("status") != expected_status
        or report.get("sealed") is not True
        or report.get("promotion_eligible") is not passed
        or report.get("partition") != request.get("partition")
        or report.get("request_hash") != hash_bytes(request_path.read_bytes())
        or report.get("finish_plan_actor_hash") != actor.actor_hash
        or report.get("matched_exam_report_file_hash") != hash_bytes(matched_path.read_bytes())
        or report.get("matched_exam_report_hash") != matched.get("report_hash")
        or report.get("metrics") != metrics
        or report.get("gates") != gates
        or report.get("implementation_hash") != _implementation_hash()
        or not isinstance(boundary, dict)
        or boundary.get("whole_body_g1_count") != 3
        or boundary.get("local_perturbation_scope_only") is not True
        or boundary.get("runtime_feature_partition_fresh") is not True
        or boundary.get("minimum_normalized_training_distance")
        != feature_partition["minimum_normalized_training_distance"]
        or boundary.get("one_shared_solver_and_ball") is not True
        or boundary.get("joint_torque_owned_only_by_frozen_neural_actor") is not True
        or boundary.get("pixels_used_for_scoring") is not False
        or boundary.get("physics_authority") != "CPU_MUJOCO"
        or boundary.get("activation_ceiling") != "SIM_ONLY"
        or boundary.get("hardware_command_sent") is not False
    ):
        raise ValueError("runtime finish plan precision exam authority is invalid")
    return report


def _derive_metrics_and_gates(
    matched: dict[str, Any], config: RuntimeFinishPlanPrecisionExamConfig
) -> tuple[dict[str, Any], dict[str, bool]]:
    rows = cast(list[dict[str, Any]], matched["rows"])
    strict = [row for row in rows if row["candidate"]["quality"]["strict_chain_passed"]]
    strict_goals = [row for row in strict if row["candidate"]["result"]["goal_crossed"]]
    precise_goals = [
        row
        for row in strict_goals
        if _finite_at_most(
            row["candidate"]["result"].get("target_error_m"), config.precision_radius_m
        )
    ]
    goal_errors = [
        float(value)
        for row in strict_goals
        if isinstance((value := row["candidate"]["result"].get("target_error_m")), int | float)
        and math.isfinite(float(value))
    ]
    base_metrics = cast(dict[str, Any], matched["metrics"])
    metrics: dict[str, Any] = {
        "case_count": len(rows),
        "candidate_strict_success_count": len(strict),
        "base_strict_success_count": int(base_metrics["base_strict_success_count"]),
        "strict_success_gain": len(strict) - int(base_metrics["base_strict_success_count"]),
        "candidate_safe_count": int(base_metrics["candidate_safe_count"]),
        "exact_replay_count": int(base_metrics["exact_replay_count"]),
        "strict_goal_count": len(strict_goals),
        "strict_save_count": sum(
            bool(row["candidate"]["result"]["goalkeeper_save_observed"]) for row in strict
        ),
        "precise_goal_count": len(precise_goals),
        "maximum_strict_goal_target_error_m": max(goal_errors, default=None),
        "mean_strict_goal_target_error_m": (
            sum(goal_errors) / len(goal_errors) if goal_errors else None
        ),
        "precision_radius_m": config.precision_radius_m,
    }
    gates = {
        "matched_exam_passed": matched.get("status") == "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT",
        "strict_success_at_least_five_of_six": len(strict) >= config.minimum_strict_successes,
        "strict_gain_at_least_two": metrics["strict_success_gain"] >= config.minimum_strict_gain,
        "candidate_safe_six_of_six": metrics["candidate_safe_count"] == 6,
        "exact_replay_six_of_six": metrics["exact_replay_count"] == 6,
        "at_least_three_precise_goals": len(precise_goals) >= config.minimum_precise_goals,
        "every_strict_goal_within_precision_radius": len(precise_goals)
        == len(strict_goals)
        == len(goal_errors),
        "at_least_one_real_goalkeeper_save": metrics["strict_save_count"] >= 1,
        "teacher_and_scripted_contact_absent": bool(
            cast(dict[str, bool], matched["gates"])["teacher_and_scripted_contact_absent"]
        ),
    }
    return metrics, gates


def _finite_at_most(value: object, maximum: float) -> bool:
    return bool(
        isinstance(value, int | float) and math.isfinite(float(value)) and float(value) <= maximum
    )


def _holdout_feature_vectors(
    cases: tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...],
    lead: DynamicLeadPassPolicy,
) -> tuple[tuple[float, ...], ...]:
    """Reconstruct the exact nine inputs observed by the runtime actor."""

    nominal_ball = (1.205, -0.160)
    vectors: list[tuple[float, ...]] = []
    for context, playmaker in cases:
        base_yaw = lead.passer_world_yaw(target_lateral_m=context.receiver_lane_m)
        passer_yaw = math.atan2(
            math.sin(base_yaw + playmaker.body_yaw_correction_rad),
            math.cos(base_yaw + playmaker.body_yaw_correction_rad),
        )
        vectors.append(
            prepared_finish_plan_features(
                receiver_lane_m=context.receiver_lane_m,
                reception_target_x_m=context.reception_target_x_m,
                passer_ball_local_xy_m=context.passer_ball_local_xy_m,
                ball_ground_friction=context.ball_ground_friction,
                passer_yaw_rad=passer_yaw,
                passer_stance_offset_xy_m=(
                    context.passer_ball_local_xy_m[0]
                    - nominal_ball[0]
                    + playmaker.stance_correction_x_m,
                    context.passer_ball_local_xy_m[1]
                    - nominal_ball[1]
                    + playmaker.stance_correction_y_m,
                ),
                passer_swing_speed_scale=playmaker.swing_speed_scale,
            )
        )
    return tuple(vectors)


def _feature_partition(
    actor: G1RuntimeFinishPlanActor,
    feature_vectors: tuple[tuple[float, ...], ...],
) -> dict[str, Any]:
    """Bind feature-level freshness, not merely renamed context identities."""

    training = tuple(
        memory.features for memory in (*actor.successful_memories, *actor.failed_memories)
    )
    scale = np.asarray(actor.feature_scale, dtype=np.float64)
    distances = [
        min(
            float(
                np.linalg.norm(
                    (np.asarray(candidate, dtype=np.float64) - np.asarray(memory)) / scale
                )
            )
            for memory in training
        )
        for candidate in feature_vectors
    ]
    return {
        "feature_vectors": [list(vector) for vector in feature_vectors],
        "feature_hashes": [hash_json(list(vector)) for vector in feature_vectors],
        "training_feature_commitment": hash_json(sorted([list(vector) for vector in training])),
        "training_feature_count": len(training),
        "minimum_normalized_training_distances": distances,
        "minimum_normalized_training_distance": min(distances),
    }


def _decode_cases(
    payload: list[object],
) -> tuple[tuple[CausalTransitionContext, PlaymakerPassProbeAction], ...]:
    decoded: list[tuple[CausalTransitionContext, PlaymakerPassProbeAction]] = []
    for value in payload:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("context"), dict)
            or not isinstance(value.get("playmaker_action"), dict)
        ):
            raise ValueError("runtime finish plan precision case is invalid")
        context = dict(value["context"])
        for field in ("passer_origin_m", "passer_ball_local_xy_m"):
            raw = context.get(field)
            if not isinstance(raw, list):
                raise ValueError("runtime finish plan precision context vector is invalid")
            context[field] = tuple(raw)
        decoded.append(
            (
                CausalTransitionContext(**context),
                PlaymakerPassProbeAction(**value["playmaker_action"]),
            )
        )
    return tuple(decoded)


def _local_artifact(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("runtime finish plan precision artifact path is invalid")
    resolved_root = root.resolve()
    artifact = (resolved_root / relative).resolve()
    if resolved_root not in artifact.parents or not artifact.is_file():
        raise ValueError("runtime finish plan precision artifact escaped evidence root")
    return artifact


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parent / "runtime_finish_plan_exam.py",
        Path(__file__).parent / "causal_transition_growth.py",
        Path(__file__).parents[1] / "growth" / "dynamic_lead_pass.py",
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
        raise ValueError("runtime finish plan precision output must be new and external")
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
    "RuntimeFinishPlanPrecisionExamConfig",
    "fresh_runtime_finish_plan_precision_holdouts",
    "run_runtime_finish_plan_precision_exam",
    "validate_runtime_finish_plan_precision_exam",
]
