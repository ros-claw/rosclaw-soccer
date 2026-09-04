"""Train a parent-anchored continuous finisher plan and multi-task critic."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.runtime_contact_target_actor import (
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    RUNTIME_FINISH_PLAN_CRITIC_NAMES,
    G1RuntimeFinishPlanActor,
    RuntimeFinishPlanAction,
    RuntimeFinishPlanContinuousPolicy,
    RuntimeFinishPlanCriticHead,
    load_runtime_finish_plan_actor,
    prepared_finish_plan_features,
    runtime_finish_plan_action_vector,
    save_runtime_finish_plan_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import _load_lead_policy
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.runtime_finish_plan_growth import _SOURCE_SCHEMAS, _row_action

_ACTION_SCALE = np.asarray((6.0, 0.02, 0.04, 0.04, 4.0, 0.04, 0.02, 1.0, 2.0, 1.0))
_CAUSAL_CRITIC_INPUT_INDICES = (5, 6, 7, 8, 16, 17, 18)


def train_continuous_runtime_finish_plan_actor(
    *,
    parent_finish_plan_actor_path: Path,
    base_target_actor_path: Path,
    source_s95_dir: Path,
    source_report_paths: tuple[Path, ...],
    output_dir: Path,
    kernel_bandwidth: float = 4.0,
    ridge_penalty: float = 0.1,
    feedback_exam_paths: tuple[Path, ...] = (),
    feedback_actor_paths: tuple[Path, ...] = (),
) -> tuple[G1RuntimeFinishPlanActor, dict[str, Any]]:
    """Fit four context-held-out kernel critics over complete physical outcomes."""

    parent_path = parent_finish_plan_actor_path.expanduser().resolve()
    parent = load_runtime_finish_plan_actor(parent_path)
    if parent.continuous_policy is not None:
        raise ValueError("continuous finish plan growth requires a discrete parent actor")
    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    if (
        parent.body_hash != base.body_hash
        or parent.kick_prior_hash != base.kick_prior_hash
        or parent.roster_hash != base.roster_hash
        or parent.finisher_self_model_hash != base.finisher_self_model_hash
        or parent.neural_contact_actor_hash != base.neural_contact_actor_hash
    ):
        raise ValueError("continuous finish plan parent lineage changed")
    if len(source_report_paths) < 4:
        raise ValueError("continuous finish plan critic needs heterogeneous outcome sources")
    if len(feedback_exam_paths) != len(feedback_actor_paths):
        raise ValueError("continuous finish plan feedback exams and actors must be paired")
    lead, lead_source = _load_lead_policy(source_s95_dir.expanduser().resolve())
    reports = tuple(
        (path.expanduser().resolve(), _bound_json(path.expanduser().resolve()))
        for path in source_report_paths
    )
    report_hashes = tuple(str(report.get("report_hash")) for _, report in reports)
    if report_hashes != parent.source_evidence_hashes:
        raise ValueError("continuous finish plan source order or commitments changed")

    parent_by_trajectory = {
        memory.trajectory_hash: (memory, strict)
        for strict, collection in (
            (True, parent.successful_memories),
            (False, parent.failed_memories),
        )
        for memory in collection
    }
    if len(parent_by_trajectory) != len(parent.successful_memories) + len(parent.failed_memories):
        raise ValueError("continuous finish plan parent contains duplicate trajectories")

    feature_rows: list[tuple[float, ...]] = []
    actions: list[RuntimeFinishPlanAction] = []
    targets: list[tuple[float, ...]] = []
    context_hashes: list[str] = []
    trajectory_hashes: list[str] = []
    continuous_feedback_rows: list[bool] = []
    rejected_continuous_feedback_rows: list[bool] = []
    source_rows: list[dict[str, Any]] = []
    for report_path, report in reports:
        schema = str(report.get("schema_version"))
        request_path = report_path.parent / "request.json"
        rows = report.get("rows")
        if (
            schema not in _SOURCE_SCHEMAS
            or report.get("promotion_eligible") is not False
            or report.get("body_hash") != parent.body_hash
            or report.get("kick_prior_hash") != parent.kick_prior_hash
            or report.get("neural_actor_hash", report.get("frozen_neural_actor_hash"))
            != parent.neural_contact_actor_hash
            or report.get("roster_hash") != parent.roster_hash
            or report.get("finisher_self_model_hash") != parent.finisher_self_model_hash
            or report.get("teacher_enabled") is not False
            or report.get("scripted_contact_torque_enabled") is not False
            or report.get("physics_authority") != "CPU_MUJOCO"
            or report.get("hardware_command_sent") is not False
            or not request_path.is_file()
            or hash_bytes(request_path.read_bytes()) != report.get("request_hash")
            or not isinstance(rows, list)
        ):
            raise ValueError("continuous finish plan source authority is invalid")
        source_rows.append(
            {
                "file": str(report_path),
                "file_hash": hash_bytes(report_path.read_bytes()),
                "report_hash": report["report_hash"],
                "schema_version": schema,
            }
        )
        for row in cast(list[dict[str, Any]], rows):
            action = _row_action(
                schema=schema,
                row=row,
                base_receive=base.required_receive_action,
            )
            if action is None:
                continue
            artifact = cast(dict[str, Any], row["trajectory"])
            trajectory_path = report_path.parent / str(artifact["file"])
            trajectory_hash = str(artifact["trajectory_digest"])
            if (
                not trajectory_path.is_file()
                or hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]
                or trajectory_hash in trajectory_hashes
            ):
                raise ValueError("continuous finish plan trajectory binding changed")
            context = cast(dict[str, Any], row["context"])
            playmaker = cast(dict[str, Any], row["playmaker_action"])
            base_yaw = lead.passer_world_yaw(target_lateral_m=float(context["receiver_lane_m"]))
            yaw = math.atan2(
                math.sin(base_yaw + float(playmaker["body_yaw_correction_rad"])),
                math.cos(base_yaw + float(playmaker["body_yaw_correction_rad"])),
            )
            features = prepared_finish_plan_features(
                receiver_lane_m=float(context["receiver_lane_m"]),
                reception_target_x_m=float(context["reception_target_x_m"]),
                passer_ball_local_xy_m=context["passer_ball_local_xy_m"],
                ball_ground_friction=float(context["ball_ground_friction"]),
                passer_yaw_rad=yaw,
                passer_stance_offset_xy_m=(
                    float(context["passer_ball_local_xy_m"][0])
                    - 1.205
                    + float(playmaker["stance_correction_x_m"]),
                    float(context["passer_ball_local_xy_m"][1])
                    + 0.160
                    + float(playmaker["stance_correction_y_m"]),
                ),
                passer_swing_speed_scale=float(playmaker["swing_speed_scale"]),
            )
            quality = cast(dict[str, Any], row["quality"])
            result = cast(dict[str, Any], row["result"])
            strict = bool(quality["strict_chain_passed"])
            prior = parent_by_trajectory.get(trajectory_hash)
            if (
                prior is None
                or prior[0].context_hash != row["context_hash"]
                or prior[0].features != features
                or prior[0].action != action
                or prior[1] is not strict
            ):
                raise ValueError("continuous finish plan parent memory binding changed")
            feature_rows.append(features)
            actions.append(action)
            targets.append(_critic_targets(quality, result))
            context_hashes.append(str(row["context_hash"]))
            trajectory_hashes.append(trajectory_hash)
            continuous_feedback_rows.append(False)
            rejected_continuous_feedback_rows.append(False)
    if set(trajectory_hashes) != set(parent_by_trajectory):
        raise ValueError("continuous finish plan source memory coverage is incomplete")
    feedback_rows: list[dict[str, Any]] = []
    feedback_evidence_hashes: list[str] = []
    for raw_exam_path, raw_actor_path in zip(
        feedback_exam_paths, feedback_actor_paths, strict=True
    ):
        exam_path = raw_exam_path.expanduser().resolve()
        feedback_actor_path = raw_actor_path.expanduser().resolve()
        exam = _bound_json(exam_path)
        feedback_actor = load_runtime_finish_plan_actor(feedback_actor_path)
        request_path = exam_path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        rows = exam.get("rows")
        if (
            exam.get("schema_version") != "rosclaw_soccer.runtime_finish_plan_exam.v1"
            or exam.get("sealed") is not False
            or exam.get("promotion_eligible") is not False
            or exam.get("finish_plan_actor_hash") != feedback_actor.actor_hash
            or feedback_actor.continuous_policy is None
            or feedback_actor.continuous_policy.parent_actor_hash != parent.actor_hash
            or not isinstance(request, dict)
            or hash_bytes(request_path.read_bytes()) != exam.get("request_hash")
            or request.get("finish_plan_actor_hash") != feedback_actor.actor_hash
            or request.get("finish_plan_actor_file_hash")
            != hash_bytes(feedback_actor_path.read_bytes())
            or request.get("physics_authority") != "CPU_MUJOCO"
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or not isinstance(rows, list)
        ):
            raise ValueError("continuous finish plan feedback authority is invalid")
        feedback_evidence_hashes.append(str(exam["report_hash"]))
        feedback_batch_passed = bool(
            exam.get("status") == "PASS_RUNTIME_FINISH_PLAN_DEVELOPMENT"
            and _feedback_preserves_direct_parent(cast(list[dict[str, Any]], rows))
        )
        feedback_rows.append(
            {
                "file": str(exam_path),
                "file_hash": hash_bytes(exam_path.read_bytes()),
                "report_hash": exam["report_hash"],
                "actor_hash": feedback_actor.actor_hash,
            }
        )
        for index, row in enumerate(cast(list[dict[str, Any]], rows)):
            candidate = cast(dict[str, Any], row["candidate"])
            artifact = cast(dict[str, Any], row["candidate_artifact"])
            replay_artifact = cast(dict[str, Any], row["replay_artifact"])
            trajectory_path = exam_path.parent / f"case-{index:03d}" / str(artifact["file"])
            replay_path = exam_path.parent / f"case-{index:03d}" / str(replay_artifact["file"])
            if (
                row.get("exact_replay") is not True
                or not trajectory_path.is_file()
                or not replay_path.is_file()
                or hash_bytes(trajectory_path.read_bytes()) != artifact.get("file_hash")
                or hash_bytes(replay_path.read_bytes()) != replay_artifact.get("file_hash")
                or artifact.get("trajectory_digest") != replay_artifact.get("trajectory_digest")
            ):
                raise ValueError("continuous finish plan feedback trajectory changed")
            features = _prepared_features_from_row(lead, row)
            decision = feedback_actor.decide(features)
            result = cast(dict[str, Any], candidate["result"])
            quality = cast(dict[str, Any], candidate["quality"])
            if (
                not decision.accepted
                or decision.action is None
                or result.get("shooter_runtime_finish_plan_actor_hash") != feedback_actor.actor_hash
                or result.get("shooter_runtime_finish_plan_route") != decision.route
                or result.get("shooter_runtime_contact_target_velocity_xyz_mps")
                != list(decision.action.target.target_foot_velocity_xyz_mps)
                or result.get("shooter_runtime_receive_stance_offset_y_m")
                != decision.action.receive.stance_offset_y_m
                or result.get("shooter_runtime_receive_foot_yaw_offset_rad")
                != decision.action.receive.foot_yaw_offset_rad
            ):
                raise ValueError("continuous finish plan feedback action binding changed")
            trajectory_hash = str(artifact["trajectory_digest"])
            if trajectory_hash in trajectory_hashes:
                raise ValueError("continuous finish plan feedback trajectory is duplicated")
            feature_rows.append(features)
            actions.append(decision.action)
            targets.append(_critic_targets(quality, result))
            context_hashes.append(str(row["context_hash"]))
            trajectory_hashes.append(trajectory_hash)
            continuous_feedback_rows.append(decision.used_continuous_policy)
            rejected_continuous_feedback_rows.append(
                bool(
                    decision.used_continuous_policy
                    and (not bool(quality["strict_chain_passed"]) or not feedback_batch_passed)
                )
            )

    features_matrix = np.asarray(feature_rows, dtype=np.float64)
    action_matrix = np.asarray(
        [runtime_finish_plan_action_vector(action) for action in actions], dtype=np.float64
    )
    observations = np.concatenate((features_matrix, action_matrix), axis=1)
    target_matrix = np.asarray(targets, dtype=np.float64)
    input_center = np.concatenate(
        (np.asarray(parent.feature_center, dtype=np.float64), np.mean(action_matrix, axis=0))
    )
    input_scale = np.concatenate(
        (np.asarray(parent.feature_scale, dtype=np.float64), _ACTION_SCALE)
    )
    normalized = (observations - input_center) / input_scale
    failed_continuous_inputs = tuple(
        tuple(float(value) for value in normalized[index])
        for index, rejected in enumerate(rejected_continuous_feedback_rows)
        if rejected
    )
    normalized_features = (
        features_matrix - input_center[: len(parent.feature_center)]
    ) / input_scale[: len(parent.feature_center)]
    blocked_continuous_context_features = tuple(
        sorted(
            {
                tuple(float(value) for value in normalized_features[index])
                for index, rejected in enumerate(rejected_continuous_feedback_rows)
                if rejected
            }
        )
    )
    contexts = sorted(set(context_hashes))
    if len(contexts) < 12:
        raise ValueError("continuous finish plan needs at least twelve physical contexts")
    validation_folds = _stratified_context_folds(
        tuple(context_hashes),
        target_matrix,
        fold_count=4,
    )
    critic_heads: list[RuntimeFinishPlanCriticHead] = []
    out_of_fold = np.zeros_like(target_matrix)
    balanced_score_out_of_fold = np.zeros_like(target_matrix)
    context_array = np.asarray(context_hashes)
    for validation in validation_folds:
        training_mask = ~np.isin(context_array, validation)
        validation_mask = ~training_mask
        training_inputs = normalized[training_mask][
            :, np.asarray(_CAUSAL_CRITIC_INPUT_INDICES, dtype=np.int64)
        ]
        training_targets = target_matrix[training_mask]
        coefficients, normalization_weights = _fit_context_balanced_normalized_head(
            training_inputs,
            training_targets,
            context_hashes=tuple(str(value) for value in context_array[training_mask]),
        )
        binary_class_priors = _binary_class_priors(
            training_targets,
            context_hashes=tuple(str(value) for value in context_array[training_mask]),
        )
        head = RuntimeFinishPlanCriticHead(
            training_context_hashes=tuple(sorted(set(context_array[training_mask]))),
            normalized_inputs=tuple(
                tuple(float(value) for value in row) for row in training_inputs
            ),
            coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
            input_indices=_CAUSAL_CRITIC_INPUT_INDICES,
            normalization_weights=tuple(
                tuple(float(value) for value in row) for row in normalization_weights
            ),
            binary_class_priors=binary_class_priors,
        )
        critic_heads.append(head)
        out_of_fold[validation_mask] = _predict_kernel_head(
            head,
            normalized[validation_mask],
            bandwidth=kernel_bandwidth,
        )
        balanced_score_out_of_fold[validation_mask] = _predict_kernel_head(
            head,
            normalized[validation_mask],
            bandwidth=kernel_bandwidth,
            apply_binary_priors=False,
        )

    context_weights = _context_equal_weights(tuple(context_hashes))
    metrics = _calibration_metrics(
        target_matrix,
        out_of_fold,
        sample_weights=context_weights,
    )
    balanced_score_metrics = _calibration_metrics(
        target_matrix,
        balanced_score_out_of_fold,
        sample_weights=context_weights,
    )
    row_weighted_metrics = _calibration_metrics(target_matrix, out_of_fold)
    training_class_support = {
        name: all(
            _has_binary_support(target_matrix[~np.isin(context_array, validation), index])
            for validation in validation_folds
        )
        for index, name in enumerate(RUNTIME_FINISH_PLAN_CRITIC_NAMES[:4])
    }
    gates = {
        "four_context_disjoint_heads": len(critic_heads) == 4
        and all(
            set(head.training_context_hashes).isdisjoint(validation)
            for head, validation in zip(critic_heads, validation_folds, strict=True)
        ),
        "every_control_head_training_fold_has_both_classes": all(
            training_class_support[name]
            for name in (
                "safe_probability",
                "intended_foot_probability",
                "strict_success_probability",
            )
        ),
        "safe_auroc_at_least_0_55": balanced_score_metrics["safe_probability"]["auroc"] >= 0.55,
        "intended_foot_auroc_at_least_0_55": balanced_score_metrics["intended_foot_probability"][
            "auroc"
        ]
        >= 0.55,
        "strict_success_auroc_at_least_0_60": balanced_score_metrics["strict_success_probability"][
            "auroc"
        ]
        >= 0.60,
        "precision_mae_at_most_0_30": metrics["precision_value"]["mae"] <= 0.30,
        "stability_mae_at_most_0_20": metrics["post_contact_stability_value"]["mae"] <= 0.20,
    }
    calibration_passed = all(gates.values())
    critic_snapshot: dict[str, Any] = {
        "parent_actor_hash": parent.actor_hash,
        "parent_training_snapshot_hash": parent.training_snapshot_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "source_reports": source_rows,
        "feedback_reports": feedback_rows,
        "trajectory_hashes": trajectory_hashes,
        "context_hashes": context_hashes,
        "critic_targets": [list(row) for row in targets],
        "validation_folds": validation_folds,
        "kernel_bandwidth": kernel_bandwidth,
        "ridge_penalty": ridge_penalty,
        "ridge_penalty_used": False,
        "critic_model": "CONTEXT_BALANCED_NORMALIZED_RBF",
        "binary_probability_calibration": "TRAINING_FOLD_CONTEXT_PRIOR_CORRECTION",
        "critic_input_indices": list(_CAUSAL_CRITIC_INPUT_INDICES),
        "critic_input_names": [
            (
                "feature:"
                + (
                    "passer_yaw_rad",
                    "passer_stance_offset_x_m",
                    "passer_stance_offset_y_m",
                    "passer_swing_speed_scale",
                )[index - 5]
                if index < 9
                else "action:"
                + (
                    "target_foot_velocity_x_mps",
                    "target_foot_velocity_y_mps",
                    "target_foot_velocity_z_mps",
                )[index - 16]
            )
            for index in _CAUSAL_CRITIC_INPUT_INDICES
        ],
        "action_scale": [float(value) for value in _ACTION_SCALE],
    }
    critic_training_snapshot_hash = str(hash_json(critic_snapshot))
    policy = RuntimeFinishPlanContinuousPolicy(
        parent_actor_hash=parent.actor_hash,
        parent_training_snapshot_hash=parent.training_snapshot_hash,
        critic_training_snapshot_hash=critic_training_snapshot_hash,
        input_center=tuple(float(value) for value in input_center),
        input_scale=tuple(float(value) for value in input_scale),
        critic_heads=tuple(critic_heads),
        kernel_bandwidth=kernel_bandwidth,
        ridge_penalty=ridge_penalty,
        feedback_evidence_hashes=tuple(feedback_evidence_hashes),
        failed_continuous_inputs=failed_continuous_inputs,
        blocked_continuous_context_features=blocked_continuous_context_features,
        critic_model="CONTEXT_BALANCED_NORMALIZED_RBF",
    )
    actor_snapshot = {
        "parent_actor_hash": parent.actor_hash,
        "critic_training_snapshot_hash": critic_training_snapshot_hash,
        "continuous_policy": asdict(policy),
        "calibration_gates": gates,
    }
    actor = replace(
        parent,
        training_snapshot_hash=str(hash_json(actor_snapshot)),
        continuous_policy=policy,
    )
    output = _new_external_output(output_dir)
    actor_path = output / "continuous-runtime-finish-plan-actor.json"
    save_runtime_finish_plan_actor(actor, actor_path)
    training_report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.continuous_runtime_finish_plan_training.v1",
        "status": (
            "PASS_CONTINUOUS_RUNTIME_FINISH_PLAN_CALIBRATION"
            if calibration_passed
            else "REJECTED_CONTINUOUS_RUNTIME_FINISH_PLAN_CALIBRATION"
        ),
        "promotion_eligible": False,
        "actor_hash": actor.actor_hash,
        "actor_file": actor_path.name,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "parent_actor_hash": parent.actor_hash,
        "parent_actor_file_hash": hash_bytes(parent_path.read_bytes()),
        "critic_training_snapshot_hash": critic_training_snapshot_hash,
        "source_report_hashes": list(report_hashes),
        "feedback_evidence_hashes": feedback_evidence_hashes,
        "metrics": {
            "trajectory_count": len(trajectory_hashes),
            "context_count": len(contexts),
            "strict_success_count": int(np.sum(target_matrix[:, 3] > 0.5)),
            "feedback_trajectory_count": len(feedback_rows) * 6,
            "failed_continuous_exclusion_count": len(failed_continuous_inputs),
            "blocked_continuous_context_count": len(blocked_continuous_context_features),
            "continuous_feedback_row_count": sum(continuous_feedback_rows),
            "critic_calibration": metrics,
            "balanced_score_discrimination": balanced_score_metrics,
            "row_weighted_critic_calibration": row_weighted_metrics,
            "training_class_support": training_class_support,
            "context_outcome_support": _context_outcome_support(
                tuple(context_hashes), target_matrix
            ),
        },
        "gates": gates,
        "auxiliary_diagnostics": {
            "clear_outcome_auroc_at_least_0_55": balanced_score_metrics[
                "clear_outcome_probability"
            ]["auroc"]
            >= 0.55,
            "clear_outcome_head_controls_runtime_selection": False,
        },
        "evidence_boundary": {
            "actor_authority": "BOUNDED_HIGH_LEVEL_PLAN_ONLY",
            "joint_torque_owner": "FROZEN_CONTENT_BOUND_NEURAL_CONTACT_ACTOR",
            "grouped_validation_unit": "CONTEXT_HASH",
            "training_contexts_have_equal_total_weight": True,
            "balanced_binary_scores_are_corrected_to_training_fold_priors": True,
            "critic_feature_policy": "PREDECLARED_PASSER_DYNAMICS_AND_TARGET_FOOT_VELOCITY",
            "ridge_penalty_used": False,
            "parent_fallback_is_immutable": True,
            "failed_continuous_actions_are_excluded": bool(feedback_rows),
            "failed_continuous_contexts_fall_back_to_parent": bool(
                blocked_continuous_context_features
            ),
            "uncalibrated_clear_outcome_head_is_diagnostic_only": True,
            "pixels_used_for_training": False,
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        },
    }
    training_report["report_hash"] = hash_json(training_report)
    _write_json(output / "training-report.json", training_report)
    return actor, training_report


def _feedback_preserves_direct_parent(rows: list[dict[str, Any]]) -> bool:
    """Reclassify legacy feedback with the current per-case parent contract."""

    parent_rows = [row for row in rows if isinstance(row.get("parent"), dict)]
    if not parent_rows:
        return True
    if len(parent_rows) != len(rows):
        return False
    for row in parent_rows:
        candidate = cast(dict[str, Any], row["candidate"])
        parent = cast(dict[str, Any], row["parent"])
        candidate_quality = cast(dict[str, Any], candidate["quality"])
        parent_quality = cast(dict[str, Any], parent["quality"])
        candidate_result = cast(dict[str, Any], candidate["result"])
        parent_result = cast(dict[str, Any], parent["result"])
        if bool(parent_quality["safe"]) and not bool(candidate_quality["safe"]):
            return False
        if bool(parent_quality["strict_chain_passed"]) and not bool(
            candidate_quality["strict_chain_passed"]
        ):
            return False
        if bool(parent_result["goalkeeper_save_observed"]) and not bool(
            candidate_result["goalkeeper_save_observed"]
        ):
            return False
        if bool(parent_result["goal_crossed"]):
            candidate_error = candidate_result.get("target_error_m")
            parent_error = parent_result.get("target_error_m")
            if (
                not bool(candidate_result["goal_crossed"])
                or not isinstance(candidate_error, int | float)
                or isinstance(candidate_error, bool)
                or not math.isfinite(float(candidate_error))
                or not isinstance(parent_error, int | float)
                or isinstance(parent_error, bool)
                or not math.isfinite(float(parent_error))
                or float(candidate_error) > float(parent_error) + 1.0e-9
            ):
                return False
    return True


def _prepared_features_from_row(lead: Any, row: dict[str, Any]) -> tuple[float, ...]:
    context = cast(dict[str, Any], row["context"])
    playmaker = cast(dict[str, Any], row["playmaker_action"])
    base_yaw = lead.passer_world_yaw(target_lateral_m=float(context["receiver_lane_m"]))
    yaw = math.atan2(
        math.sin(base_yaw + float(playmaker["body_yaw_correction_rad"])),
        math.cos(base_yaw + float(playmaker["body_yaw_correction_rad"])),
    )
    return prepared_finish_plan_features(
        receiver_lane_m=float(context["receiver_lane_m"]),
        reception_target_x_m=float(context["reception_target_x_m"]),
        passer_ball_local_xy_m=context["passer_ball_local_xy_m"],
        ball_ground_friction=float(context["ball_ground_friction"]),
        passer_yaw_rad=yaw,
        passer_stance_offset_xy_m=(
            float(context["passer_ball_local_xy_m"][0])
            - 1.205
            + float(playmaker["stance_correction_x_m"]),
            float(context["passer_ball_local_xy_m"][1])
            + 0.160
            + float(playmaker["stance_correction_y_m"]),
        ),
        passer_swing_speed_scale=float(playmaker["swing_speed_scale"]),
    )


def _critic_targets(
    quality: dict[str, Any], result: dict[str, Any]
) -> tuple[float, float, float, float, float, float]:
    strict = bool(quality["strict_chain_passed"])
    target_error = result.get("target_error_m")
    if (
        strict
        and result.get("goal_crossed") is True
        and isinstance(target_error, int | float)
        and not isinstance(target_error, bool)
        and math.isfinite(float(target_error))
    ):
        precision = max(0.0, 1.0 - float(target_error) / 1.5)
    elif strict and result.get("goalkeeper_save_observed") is True:
        precision = 0.65
    else:
        precision = 0.0
    pelvis = float(result.get("shooter_min_pelvis_height_m", 0.0))
    slip = float(result.get("shooter_post_contact_support_foot_slip_m", 1.0))
    roll = float(result.get("shooter_roll_peak_rad", 1.0))
    if not all(math.isfinite(value) for value in (pelvis, slip, roll)):
        stability = 0.0
    else:
        stability = float(
            np.clip((pelvis - 0.50) / 0.20, 0.0, 1.0)
            * np.clip(1.0 - slip / 0.25, 0.0, 1.0)
            * np.clip(1.0 - roll / 0.55, 0.0, 1.0)
        )
    return (
        float(bool(quality["safe"])),
        float(bool(quality["intended_foot_contact"])),
        float(bool(quality["clear_outcome"])),
        float(strict),
        precision,
        stability,
    )


def _fit_weighted_kernel_head(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    context_hashes: tuple[str, ...],
    bandwidth: float,
    ridge_penalty: float,
) -> np.ndarray:
    if len(context_hashes) != len(inputs):
        raise ValueError("continuous critic context weights are misaligned")
    distances = np.sum(np.square(inputs[:, None, :] - inputs[None, :, :]), axis=2)
    kernel = np.exp(-distances / (2.0 * bandwidth**2))
    coefficients = np.empty_like(targets)
    context_weights = _context_equal_weights(context_hashes)
    for index in range(targets.shape[1]):
        values = targets[:, index]
        positive = values > 0.0
        if index < 4 and np.any(positive) and np.any(~positive):
            class_weights = np.where(
                positive,
                1.0 / float(np.sum(context_weights[positive])),
                1.0 / float(np.sum(context_weights[~positive])),
            )
        elif index == 4 and np.any(positive):
            class_weights = np.where(positive, 2.0, 1.0)
        else:
            class_weights = np.ones_like(values)
        weights = context_weights * class_weights
        weights *= len(weights) / float(np.sum(weights))
        regularizer = ridge_penalty * np.diag(1.0 / weights)
        coefficients[:, index] = np.linalg.solve(kernel + regularizer, values)
    return coefficients


def _fit_context_balanced_normalized_head(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    context_hashes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Store weighted targets and masses for a normalized RBF smoother."""

    if (
        inputs.ndim != 2
        or targets.shape != (len(inputs), len(RUNTIME_FINISH_PLAN_CRITIC_NAMES))
        or len(context_hashes) != len(inputs)
        or not np.all(np.isfinite(inputs))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("continuous normalized critic training rows are invalid")
    numerator = np.empty_like(targets)
    normalization_weights = np.empty_like(targets)
    context_weights = _context_equal_weights(context_hashes)
    for index in range(targets.shape[1]):
        values = targets[:, index]
        positive = values > 0.0
        if index < 4 and np.any(positive) and np.any(~positive):
            class_weights = np.where(
                positive,
                1.0 / float(np.sum(context_weights[positive])),
                1.0 / float(np.sum(context_weights[~positive])),
            )
        elif index == 4 and np.any(positive):
            class_weights = np.where(positive, 2.0, 1.0)
        else:
            class_weights = np.ones_like(values)
        weights = context_weights * class_weights
        weights *= len(weights) / float(np.sum(weights))
        normalization_weights[:, index] = weights
        numerator[:, index] = weights * values
    return numerator, normalization_weights


def _binary_class_priors(
    targets: np.ndarray,
    *,
    context_hashes: tuple[str, ...],
) -> tuple[float, ...]:
    """Estimate binary priors without allowing a held-out context to leak in."""

    if targets.ndim != 2 or targets.shape[1] < 4 or len(context_hashes) != len(targets):
        raise ValueError("continuous critic prior rows are invalid")
    weights = _context_equal_weights(context_hashes)
    priors = tuple(float(np.average(targets[:, index], weights=weights)) for index in range(4))
    if any(not 0.0 < value < 1.0 for value in priors):
        raise ValueError("continuous critic training fold lacks binary prior support")
    return priors


def _predict_kernel_head(
    head: RuntimeFinishPlanCriticHead,
    inputs: np.ndarray,
    *,
    bandwidth: float,
    apply_binary_priors: bool = True,
) -> np.ndarray:
    training = np.asarray(head.normalized_inputs, dtype=np.float64)
    query = (
        inputs[:, np.asarray(head.input_indices, dtype=np.int64)] if head.input_indices else inputs
    )
    distances = np.sum(np.square(query[:, None, :] - training[None, :, :]), axis=2)
    kernel = np.exp(-distances / (2.0 * bandwidth**2))
    predictions = kernel @ np.asarray(head.coefficients)
    if head.normalization_weights:
        denominator = kernel @ np.asarray(head.normalization_weights)
        predictions = predictions / np.maximum(denominator, 1.0e-12)
    predictions = np.clip(predictions, 0.0, 1.0)
    if apply_binary_priors and head.binary_class_priors:
        priors = np.asarray(head.binary_class_priors, dtype=np.float64)
        balanced = predictions[:, :4]
        predictions[:, :4] = (balanced * priors) / np.maximum(
            balanced * priors + (1.0 - balanced) * (1.0 - priors),
            1.0e-12,
        )
    return np.asarray(np.clip(predictions, 0.0, 1.0), dtype=np.float64)


def _calibration_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    weights = (
        np.ones(targets.shape[0], dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if (
        weights.shape != (targets.shape[0],)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("continuous critic calibration weights are invalid")
    metrics: dict[str, Any] = {}
    for index, name in enumerate(RUNTIME_FINISH_PLAN_CRITIC_NAMES):
        expected = targets[:, index]
        predicted = np.clip(predictions[:, index], 0.0, 1.0)
        row: dict[str, Any] = {
            "mae": float(np.average(np.abs(predicted - expected), weights=weights)),
            "brier": float(np.average(np.square(predicted - expected), weights=weights)),
            "target_mean": float(np.average(expected, weights=weights)),
            "prediction_mean": float(np.average(predicted, weights=weights)),
        }
        if index < 4:
            row["auroc"] = _binary_auroc(expected, predicted, sample_weights=weights)
        metrics[name] = row
    return metrics


def _binary_auroc(
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
) -> float:
    weights = (
        np.ones(targets.shape[0], dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    positive = predictions[targets > 0.5]
    negative = predictions[targets <= 0.5]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    positive_weights = weights[targets > 0.5]
    negative_weights = weights[targets <= 0.5]
    comparisons = positive[:, None] - negative[None, :]
    pair_weights = positive_weights[:, None] * negative_weights[None, :]
    wins = np.asarray(comparisons > 0.0, dtype=np.float64) + 0.5 * np.asarray(
        comparisons == 0.0, dtype=np.float64
    )
    return float(np.sum(wins * pair_weights) / np.sum(pair_weights))


def _context_equal_weights(context_hashes: tuple[str, ...]) -> np.ndarray:
    if not context_hashes:
        raise ValueError("continuous critic contexts are empty")
    counts: dict[str, int] = {}
    for context_hash in context_hashes:
        counts[context_hash] = counts.get(context_hash, 0) + 1
    weights = np.asarray([1.0 / counts[value] for value in context_hashes], dtype=np.float64)
    return weights * (len(weights) / float(np.sum(weights)))


def _has_binary_support(values: np.ndarray) -> bool:
    return bool(np.any(values > 0.5) and np.any(values <= 0.5))


def _context_outcome_support(
    context_hashes: tuple[str, ...], targets: np.ndarray
) -> dict[str, dict[str, int]]:
    context_array = np.asarray(context_hashes)
    return {
        name: {
            "positive_context_count": sum(
                bool(np.any(targets[context_array == context_hash, index] > 0.5))
                for context_hash in sorted(set(context_hashes))
            ),
            "negative_context_count": sum(
                bool(np.any(targets[context_array == context_hash, index] <= 0.5))
                for context_hash in sorted(set(context_hashes))
            ),
        }
        for index, name in enumerate(RUNTIME_FINISH_PLAN_CRITIC_NAMES[:4])
    }


def _stratified_context_folds(
    context_hashes: tuple[str, ...],
    targets: np.ndarray,
    *,
    fold_count: int,
) -> tuple[tuple[str, ...], ...]:
    """Assign whole contexts while prioritizing rare safety and contact labels."""

    if fold_count != 4 or len(context_hashes) != len(targets):
        raise ValueError("continuous critic fold request is invalid")
    context_array = np.asarray(context_hashes)
    contexts = sorted(set(context_hashes))
    if len(contexts) < fold_count * 3:
        raise ValueError("continuous critic needs at least three contexts per fold")
    summaries: dict[str, np.ndarray] = {}
    for context_hash in contexts:
        rows = targets[context_array == context_hash]
        summaries[context_hash] = np.asarray(
            (
                np.sum(rows[:, 0] <= 0.5),
                np.sum(rows[:, 1] <= 0.5),
                np.sum(rows[:, 2] > 0.5),
                np.sum(rows[:, 3] > 0.5),
                len(rows),
            ),
            dtype=np.float64,
        )
    global_total = np.sum(np.asarray(list(summaries.values())), axis=0)
    normalizer = np.maximum(global_total, 1.0)
    ordered = sorted(
        contexts,
        key=lambda value: (
            summaries[value][0] > 0.0,
            summaries[value][1] > 0.0,
            summaries[value][3] > 0.0,
            summaries[value][2] > 0.0,
            float(np.sum(summaries[value] / normalizer)),
            summaries[value][4],
            value,
        ),
        reverse=True,
    )
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    totals = np.zeros((fold_count, len(global_total)), dtype=np.float64)
    for context_hash in ordered:
        summary = summaries[context_hash]
        choices: list[tuple[float, int, int]] = []
        for fold_index in range(fold_count):
            trial = totals.copy()
            trial[fold_index] += summary
            normalized = trial / normalizer
            imbalance = float(
                5.0 * np.var(normalized[:, 0])
                + 2.0 * np.var(normalized[:, 1])
                + np.var(normalized[:, 2])
                + 2.0 * np.var(normalized[:, 3])
                + 0.5 * np.var(normalized[:, 4])
            )
            choices.append((imbalance, len(folds[fold_index]), fold_index))
        _, _, selected = min(choices)
        folds[selected].append(context_hash)
        totals[selected] += summary
    result = tuple(tuple(sorted(fold)) for fold in folds)
    if any(len(fold) < 3 for fold in result) or set().union(*map(set, result)) != set(contexts):
        raise ValueError("continuous critic stratified folds are incomplete")
    return result


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("continuous finish plan output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_continuous_runtime_finish_plan_actor"]
