from __future__ import annotations

import math
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
from rosclaw_soccer.training.continuous_finish_plan_growth import _critic_targets
from rosclaw_soccer.training.continuous_finish_plan_repair import (
    _local_refinement_actions,
    _micro_refinement_actions,
)
from rosclaw_soccer.training.runtime_finish_plan_growth import _row_action

_CURRENT_PARENT = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s169-prepared-finish-plan-actor-v4/runtime-finish-plan-actor.json"
)
_CURRENT_GROWN = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s174-prepared-finish-plan-actor-v5-continuous-memory/runtime-finish-plan-actor.json"
)
_CURRENT_CONTINUOUS = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s171-continuous-finish-plan-critic-v2/continuous-runtime-finish-plan-actor.json"
)
_CURRENT_DEVELOPMENT = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s175-prepared-finish-plan-v5-consumed-development-v2/exam-report.json"
)
_CURRENT_VIDEO = Path(
    "/code/rosclaw/rosclaw_football/videos/s175-continuous-finish-plan-development-v2.json"
)


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


def _continuous_actor(*, exclude_candidate: bool = False) -> G1RuntimeFinishPlanActor:
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
        feedback_evidence_hashes=("sha256:" + "4" * 64,) if exclude_candidate else (),
        failed_continuous_inputs=(tuple(float(value) for value in candidate_input),)
        if exclude_candidate
        else (),
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

    assert len(local) >= 8
    assert len(micro) >= 8
    assert len({action.action_hash for action in local}) == len(local)
    assert len({action.action_hash for action in micro}) == len(micro)
    assert all(not action.direct_joint_torque_output for action in (*local, *micro))

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


def test_current_growth_artifacts_load_when_available() -> None:
    if not all(
        path.is_file()
        for path in (
            _CURRENT_PARENT,
            _CURRENT_GROWN,
            _CURRENT_CONTINUOUS,
            _CURRENT_DEVELOPMENT,
            _CURRENT_VIDEO,
        )
    ):
        pytest.skip("current continuous finish plan evidence is unavailable")

    parent = load_runtime_finish_plan_actor(_CURRENT_PARENT)
    grown = load_runtime_finish_plan_actor(_CURRENT_GROWN)
    continuous = load_runtime_finish_plan_actor(_CURRENT_CONTINUOUS)
    video = validate_continuous_finish_plan_development_video(_CURRENT_VIDEO)

    assert (
        parent.actor_hash
        == "sha256:a164982fbd8413dbd20016b965a4760f51c3e61cd3b23204d37660b5d7614d0e"
    )
    assert grown.continuous_policy is None
    assert continuous.continuous_policy is not None
    assert continuous.continuous_policy.parent_actor_hash == parent.actor_hash
    assert len(grown.successful_memories) == 44
    assert len(grown.failed_memories) == 254
    assert {memory.trajectory_hash for memory in parent.successful_memories}.issubset(
        memory.trajectory_hash for memory in grown.successful_memories
    )
    assert video["source_exam_hash"] == (
        "sha256:d33f0229207138fd31a2336ce247eedde62ccda5c8a61052e08e7b7d1ae2086c"
    )
    assert video["fresh_generalization_claimed"] is False
