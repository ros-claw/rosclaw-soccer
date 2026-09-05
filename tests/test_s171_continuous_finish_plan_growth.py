from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.runtime_contact_target_actor import RuntimeContactTargetAction
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    G1RuntimeFinishPlanActor,
    RuntimeFinishPlanAction,
    RuntimeFinishPlanContinuousPolicy,
    RuntimeFinishPlanCriticHead,
    RuntimeFinishPlanMemory,
    load_runtime_finish_plan_actor,
    runtime_finish_plan_action_from_vector,
    runtime_finish_plan_action_vector,
    save_runtime_finish_plan_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.media.continuous_finish_plan_development_video import (
    validate_continuous_finish_plan_development_video,
)
from rosclaw_soccer.training.continuous_finish_plan_growth import (
    _binary_class_priors,
    _context_equal_weights,
    _critic_targets,
    _feedback_preserves_direct_parent,
    _fit_context_balanced_normalized_head,
    _stratified_context_folds,
)
from rosclaw_soccer.training.continuous_finish_plan_repair import (
    _local_refinement_actions,
    _micro_refinement_actions,
    _stance_coverage_actions,
)
from rosclaw_soccer.training.runtime_finish_plan_growth import _row_action


def _action(*, stance_x: float, stance_y: float, target_y: float) -> RuntimeFinishPlanAction:
    return RuntimeFinishPlanAction(
        receive=RuntimeReceiveAction(
            maximum_arrival_advance_frames=18,
            stance_offset_x_m=stance_x,
            stance_offset_y_m=stance_y,
            contact_policy_frame=252,
            foot_yaw_offset_rad=-0.08,
            foot_pitch_offset_rad=0.01,
        ),
        target=RuntimeContactTargetAction((9.0, target_y, -1.0)),
    )


def _memory(index: int, feature: float, action: RuntimeFinishPlanAction) -> RuntimeFinishPlanMemory:
    return RuntimeFinishPlanMemory(
        context_hash="sha256:" + format(index, "x") * 64,
        trajectory_hash="sha256:" + "0" * 62 + f"{index:02x}",
        features=(feature, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action=action,
        quality_score=10.0,
    )


def _parent() -> G1RuntimeFinishPlanActor:
    early = _action(stance_x=-0.08, stance_y=-0.06, target_y=5.0)
    late = _action(stance_x=0.08, stance_y=0.06, target_y=1.0)
    return G1RuntimeFinishPlanActor(
        body_hash="sha256:" + "a" * 64,
        kick_prior_hash="sha256:" + "b" * 64,
        roster_hash="sha256:" + "c" * 64,
        finisher_self_model_hash="sha256:" + "d" * 64,
        neural_contact_actor_hash="sha256:" + "e" * 64,
        contact_handoff_actor_hash="sha256:" + "2" * 64,
        contact_handoff_offset_frames=7,
        source_evidence_hashes=("sha256:" + "f" * 64,),
        training_snapshot_hash="sha256:" + "1" * 64,
        feature_center=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        successful_memories=(
            _memory(1, -0.02, early),
            _memory(2, -0.03, early),
            _memory(3, -0.02, late),
            _memory(4, -0.03, late),
        ),
        failed_memories=tuple(
            _memory(index, 2.0 + index, early if index < 9 else late) for index in range(5, 13)
        ),
    )


def _continuous_actor(
    *, exclude_candidate: bool = False, block_context: bool = False
) -> G1RuntimeFinishPlanActor:
    parent = _parent()
    features = np.asarray((-0.025,) + (0.0,) * 8)
    ranked = sorted(
        parent.successful_memories,
        key=lambda memory: abs(memory.features[0] - features[0]),
    )
    centroid = np.mean(
        [runtime_finish_plan_action_vector(memory.action) for memory in ranked], axis=0
    )
    parent_action = parent.decide(features).action
    assert parent_action is not None
    candidate = runtime_finish_plan_action_from_vector(centroid)
    center = np.concatenate((features, runtime_finish_plan_action_vector(parent_action)))
    scale = np.ones(19)
    candidate_input = (
        np.concatenate((features, runtime_finish_plan_action_vector(candidate))) - center
    ) / scale
    inputs = np.full((32, 19), 25.0)
    inputs[0] = 0.0
    inputs[1] = candidate_input
    targets = np.asarray(
        (
            (1.0, 0.8, 0.5, 0.2, 0.1, 0.5),
            (1.0, 0.8, 0.5, 0.8, 0.7, 0.5),
        )
    )
    distance = float(np.sum(np.square(candidate_input)))
    kernel = np.asarray(((1.0, math.exp(-distance / 0.5)), (math.exp(-distance / 0.5), 1.0)))
    coefficients = np.zeros((32, 6))
    coefficients[:2] = np.linalg.solve(kernel, targets)
    head = RuntimeFinishPlanCriticHead(
        training_context_hashes=tuple("sha256:" + format(i, "x") * 64 for i in range(1, 6)),
        normalized_inputs=tuple(tuple(float(value) for value in row) for row in inputs),
        coefficients=tuple(tuple(float(value) for value in row) for row in coefficients),
    )
    policy = RuntimeFinishPlanContinuousPolicy(
        parent_actor_hash=parent.actor_hash,
        parent_training_snapshot_hash=parent.training_snapshot_hash,
        critic_training_snapshot_hash="sha256:" + "3" * 64,
        input_center=tuple(float(value) for value in center),
        input_scale=tuple(float(value) for value in scale),
        critic_heads=(head, head, head, head),
        kernel_bandwidth=0.5,
        nearest_success_count=4,
        interpolation_bandwidth=4.0,
        interpolation_alphas=(1.0,),
        maximum_ensemble_spread=0.01,
        feedback_evidence_hashes=("sha256:" + "4" * 64,)
        if exclude_candidate or block_context
        else (),
        failed_continuous_inputs=(tuple(float(value) for value in candidate_input),)
        if exclude_candidate
        else (),
        blocked_continuous_context_features=((0.0,) * 9,) if block_context else (),
    )
    return replace(
        parent,
        training_snapshot_hash="sha256:" + "5" * 64,
        continuous_policy=policy,
    )


def test_continuous_plan_uses_critic_but_keeps_parent_fallback() -> None:
    features = (-0.025,) + (0.0,) * 8
    actor = _continuous_actor()
    decision = actor.decide(features)

    assert decision.accepted
    assert decision.used_continuous_policy
    assert decision.route == "VERIFIED_RUNTIME_CONTINUOUS_FINISH_PLAN"
    assert decision.parent_action_hash is not None
    assert decision.critic_strict_mean == pytest.approx(0.8)
    assert decision.critic_precision_mean == pytest.approx(0.7)
    assert decision.action is not None and not decision.action.direct_joint_torque_output

    excluded = _continuous_actor(exclude_candidate=True).decide(features)
    assert excluded.accepted
    assert not excluded.used_continuous_policy
    assert excluded.route == "VERIFIED_RUNTIME_FINISH_PLAN"
    assert excluded.parent_action_hash is not None

    blocked = _continuous_actor(block_context=True).decide(features)
    assert blocked.accepted
    assert not blocked.used_continuous_policy
    assert blocked.route == "VERIFIED_RUNTIME_FINISH_PLAN"
    assert blocked.parent_action_hash is not None


def test_continuous_actor_round_trip_and_parent_commitment(tmp_path: Path) -> None:
    actor = _continuous_actor(exclude_candidate=True)
    path = tmp_path / "continuous.json"
    save_runtime_finish_plan_actor(actor, path)

    assert load_runtime_finish_plan_actor(path) == actor
    payload = path.read_text(encoding="utf-8")
    path.write_text(
        payload.replace(actor.continuous_policy.parent_actor_hash, "sha256:" + "9" * 64)
    )
    with pytest.raises(ValueError, match="hash mismatch|parent binding"):
        load_runtime_finish_plan_actor(path)


def test_normalized_critic_uses_selected_inputs_and_context_balanced_mass() -> None:
    inputs = np.asarray([[float(index % 2)] for index in range(32)])
    targets = np.zeros((32, 6), dtype=np.float64)
    targets[1::2] = 1.0
    contexts = tuple(f"context-{index // 8}" for index in range(32))
    numerator, weights = _fit_context_balanced_normalized_head(
        inputs,
        targets,
        context_hashes=contexts,
    )
    head = RuntimeFinishPlanCriticHead(
        training_context_hashes=tuple("sha256:" + str(index) * 64 for index in range(1, 5)),
        normalized_inputs=tuple(tuple(row) for row in inputs),
        coefficients=tuple(tuple(row) for row in numerator),
        input_indices=(17,),
        normalization_weights=tuple(tuple(row) for row in weights),
        binary_class_priors=(0.5,) * 4,
    )
    parent = _parent()
    policy = RuntimeFinishPlanContinuousPolicy(
        parent_actor_hash=parent.actor_hash,
        parent_training_snapshot_hash=parent.training_snapshot_hash,
        critic_training_snapshot_hash="sha256:" + "3" * 64,
        input_center=(0.0,) * 19,
        input_scale=(1.0,) * 19,
        critic_heads=(head,) * 4,
        kernel_bandwidth=0.5,
        critic_model="CONTEXT_BALANCED_NORMALIZED_RBF",
    )

    low = policy.predict((0.0,) * 9, _action(stance_x=0.0, stance_y=0.0, target_y=0.0))
    high = policy.predict((0.0,) * 9, _action(stance_x=0.0, stance_y=0.0, target_y=1.0))

    assert np.all(low < high)
    assert np.all(weights > 0.0)


def test_binary_priors_use_context_equal_mass() -> None:
    targets = np.zeros((5, 6), dtype=np.float64)
    targets[:4, :4] = 1.0

    priors = _binary_class_priors(
        targets,
        context_hashes=("many", "many", "many", "many", "single"),
    )

    assert priors == pytest.approx((0.5,) * 4)


def test_feedback_reclassifies_per_case_parent_precision_regression() -> None:
    rows = [
        {
            "candidate": {
                "quality": {"safe": True, "strict_chain_passed": True},
                "result": {
                    "goal_crossed": True,
                    "goalkeeper_save_observed": False,
                    "target_error_m": 0.40,
                },
            },
            "parent": {
                "quality": {"safe": True, "strict_chain_passed": True},
                "result": {
                    "goal_crossed": True,
                    "goalkeeper_save_observed": False,
                    "target_error_m": 0.10,
                },
            },
        }
    ]

    assert not _feedback_preserves_direct_parent(rows)


def test_action_vector_decoder_reapplies_every_sim_only_bound() -> None:
    action = runtime_finish_plan_action_from_vector(
        (17.0, -1.0, -2.0, 2.0, 999.0, -2.0, 2.0, 99.0, -99.0, 99.0)
    )

    assert action.receive.maximum_arrival_advance_frames == 18
    assert action.receive.arrival_alignment_tolerance_sec == 0.02
    assert action.receive.stance_offset_x_m == -0.12
    assert action.receive.stance_offset_y_m == 0.12
    assert action.receive.contact_policy_frame == 258
    assert action.target.target_foot_velocity_xyz_mps == (12.0, -6.0, 6.0)
    assert not action.direct_joint_torque_output


def test_continuous_repair_rows_restore_actions_and_build_bounded_neighborhoods() -> None:
    parent = _action(stance_x=0.12, stance_y=0.12, target_y=0.0)
    local = _local_refinement_actions(parent)
    micro = _micro_refinement_actions((parent,))
    coverage = _stance_coverage_actions((parent,))

    assert len(local) >= 8
    assert len(micro) >= 8
    assert len(coverage) >= 12
    assert len({action.action_hash for action in local}) == len(local)
    assert len({action.action_hash for action in micro}) == len(micro)
    assert all(not action.direct_joint_torque_output for action in (*local, *micro))
    assert {-0.12, 0.12}.issubset(action.receive.stance_offset_y_m for action in coverage)

    restored = _row_action(
        schema="rosclaw_soccer.continuous_finish_plan_repair.v1",
        row={"action": {"receive": parent.receive.__dict__, "target": parent.target.__dict__}},
        base_receive=RuntimeReceiveAction(),
    )
    assert restored == parent


def test_multitask_targets_penalize_unsafe_or_imprecise_outcomes() -> None:
    good = _critic_targets(
        {
            "safe": True,
            "intended_foot_contact": True,
            "clear_outcome": True,
            "strict_chain_passed": True,
        },
        {
            "goal_crossed": True,
            "goalkeeper_save_observed": False,
            "target_error_m": 0.05,
            "shooter_min_pelvis_height_m": 0.70,
            "shooter_post_contact_support_foot_slip_m": 0.02,
            "shooter_roll_peak_rad": 0.20,
        },
    )
    bad = _critic_targets(
        {
            "safe": False,
            "intended_foot_contact": False,
            "clear_outcome": False,
            "strict_chain_passed": False,
        },
        {
            "goal_crossed": False,
            "goalkeeper_save_observed": False,
            "target_error_m": None,
            "shooter_min_pelvis_height_m": 0.40,
            "shooter_post_contact_support_foot_slip_m": 0.40,
            "shooter_roll_peak_rad": 0.70,
        },
    )

    assert good[:4] == (1.0, 1.0, 1.0, 1.0)
    assert good[4] > 0.9 and good[5] > 0.0
    assert bad == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_context_balancing_gives_each_physics_context_equal_mass() -> None:
    contexts = ("many", "many", "many", "many", "single")
    weights = _context_equal_weights(contexts)

    assert np.sum(weights[np.asarray(contexts) == "many"]) == pytest.approx(
        np.sum(weights[np.asarray(contexts) == "single"])
    )
    assert np.mean(weights) == pytest.approx(1.0)


def test_stratified_folds_keep_rare_unsafe_contexts_disjoint_and_balanced() -> None:
    contexts = tuple(f"context-{index:02d}" for index in range(16) for _ in range(2))
    targets = np.ones((len(contexts), 6), dtype=np.float64)
    unsafe_contexts = {"context-00", "context-04", "context-08", "context-12"}
    for row_index, context in enumerate(contexts):
        targets[row_index, 0] = 0.0 if context in unsafe_contexts else 1.0
        targets[row_index, 1] = 0.0 if int(context[-2:]) % 2 else 1.0
        targets[row_index, 2] = float(int(context[-2:]) % 3 == 0)
        targets[row_index, 3] = float(int(context[-2:]) % 4 == 0)

    folds = _stratified_context_folds(contexts, targets, fold_count=4)

    assert set().union(*map(set, folds)) == set(contexts)
    assert sum(len(fold) for fold in folds) == 16
    assert all(len(set(fold) & unsafe_contexts) == 1 for fold in folds)


def test_current_growth_artifacts_load_when_available() -> None:
    evidence_value = os.environ.get("ROSCLAW_SOCCER_EVIDENCE")
    video_value = os.environ.get("ROSCLAW_SOCCER_VIDEO")
    if evidence_value is None or video_value is None:
        pytest.skip("external soccer evidence roots are not configured")
    evidence = Path(evidence_value).expanduser().resolve()
    video_root = Path(video_value).expanduser().resolve()
    parent_path = (
        evidence / "s189-finish-plan-actor-v6-precision-credit-v2/runtime-finish-plan-actor.json"
    )
    continuous_path = (
        evidence / "s195-context-balanced-continuous-critic-v3-context-fuse/"
        "continuous-runtime-finish-plan-actor.json"
    )
    development_path = (
        evidence / "s197-continuous-finish-plan-v3-final-parent-safe-development/exam-report.json"
    )
    video_path = video_root / "s197-context-safe-continuous-growth-final-development.json"
    if not all(
        path.is_file() for path in (parent_path, continuous_path, development_path, video_path)
    ):
        pytest.skip("current continuous finish plan evidence is unavailable")

    parent = load_runtime_finish_plan_actor(parent_path)
    continuous = load_runtime_finish_plan_actor(continuous_path)
    video = validate_continuous_finish_plan_development_video(video_path)

    assert (
        parent.actor_hash
        == "sha256:8792384c44740c6049b98e6a7d3ca9079a7998e7501f58a8d75eb5f75e19690f"
    )
    assert continuous.continuous_policy is not None
    assert continuous.continuous_policy.parent_actor_hash == parent.actor_hash
    assert len(parent.successful_memories) == 66
    assert len(parent.failed_memories) == 279
    assert len(continuous.continuous_policy.blocked_continuous_context_features) == 2
    assert (
        video["source_exam_hash"]
        == json.loads(development_path.read_text(encoding="utf-8"))["report_hash"]
    )
    assert video["source_candidate_strict_success_count"] == 6
    assert video["source_candidate_precise_goal_count"] == 3
    assert video["fresh_generalization_claimed"] is False
