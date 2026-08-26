from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest
from rosclaw.continual import (
    GrowthSafetyObservation,
    SuccessorStateSample,
    SuccessorStateTracker,
    evaluate_growth_safety,
)

from rosclaw_soccer.training.recovery_foundation import (
    ProprioceptiveRecoveryGate,
    RecoveryExpert,
    RecoveryGateObservation,
    RecoveryProprioFrame,
    RecoveryResetSource,
    RecoveryTrainingDistribution,
    blend_recovery_actions,
    build_controlled_fall_safety_profiles,
    build_goalkeeper_ready_successor_contract,
)

_HASH = "sha256:" + "a" * 64


def _frame(
    *,
    gravity: float,
    height: float,
    linear: float,
    angular: float,
    support: float,
    nonfoot: float,
    joint_speed: float = 0.2,
) -> RecoveryProprioFrame:
    return RecoveryProprioFrame(
        projected_gravity_z=gravity,
        pelvis_height_m=height,
        root_linear_speed_mps=linear,
        root_angular_speed_rad_s=angular,
        left_foot_load_normalized=support,
        right_foot_load_normalized=support,
        nonfoot_contact_load_normalized=nonfoot,
        mean_joint_speed_rad_s=joint_speed,
        previous_action_delta_rms=0.05,
    )


@pytest.mark.parametrize(
    ("frame", "expected"),
    (
        (
            _frame(
                gravity=0.2,
                height=0.55,
                linear=2.0,
                angular=4.0,
                support=0.0,
                nonfoot=0.5,
                joint_speed=2.0,
            ),
            RecoveryExpert.ABSORB,
        ),
        (
            _frame(
                gravity=-0.9,
                height=0.20,
                linear=0.05,
                angular=0.1,
                support=0.0,
                nonfoot=0.8,
            ),
            RecoveryExpert.GET_UP,
        ),
        (
            _frame(
                gravity=1.0,
                height=0.78,
                linear=0.05,
                angular=0.1,
                support=0.9,
                nonfoot=0.0,
            ),
            RecoveryExpert.ATHLETE,
        ),
    ),
)
def test_proprioceptive_gate_routes_softly_without_privileged_stage(
    frame: RecoveryProprioFrame,
    expected: RecoveryExpert,
) -> None:
    forbidden = {"stage", "reference_phase", "expert_label", "task_truth"}
    assert forbidden.isdisjoint(field.name for field in fields(RecoveryGateObservation))
    gate = ProprioceptiveRecoveryGate()
    output = gate.infer(RecoveryGateObservation(history=(frame,) * 4))
    assert output.dominant_expert is expected
    assert sum(output.expert_weights.values()) == pytest.approx(1.0)
    assert all(0.0 < value < 1.0 for value in output.expert_weights.values())


def test_recovery_action_blend_is_full_body_and_authority_bounded() -> None:
    frame = _frame(
        gravity=1.0,
        height=0.78,
        linear=0.0,
        angular=0.0,
        support=1.0,
        nonfoot=0.0,
    )
    output = ProprioceptiveRecoveryGate().infer(RecoveryGateObservation(history=(frame,) * 4))
    actions = {
        RecoveryExpert.ABSORB: np.full(29, -0.8),
        RecoveryExpert.GET_UP: np.zeros(29),
        RecoveryExpert.ATHLETE: np.full(29, 0.6),
    }
    blended = blend_recovery_actions(actions=actions, gate=output)
    assert blended.shape == (29,)
    assert np.isfinite(blended).all()
    assert np.max(np.abs(blended)) <= 1.0
    with pytest.raises(ValueError, match="exceeds"):
        blend_recovery_actions(
            actions={**actions, RecoveryExpert.ATHLETE: np.full(29, 1.01)},
            gate=output,
        )


def test_recovery_training_distribution_matches_frozen_campaign() -> None:
    allocation = RecoveryTrainingDistribution().allocate(100)
    assert allocation == {
        RecoveryResetSource.TRUE_POST_SKILL: 20,
        RecoveryResetSource.PHYSICS_PERTURBATION: 30,
        RecoveryResetSource.RANDOMIZED_RESET: 20,
        RecoveryResetSource.DIVE_INTERMEDIATE: 15,
        RecoveryResetSource.HARDEST_FAILURE_MEMORY: 10,
        RecoveryResetSource.NIGHTMARE: 5,
    }


def test_goalkeeper_ready_requires_one_continuous_second() -> None:
    contract = build_goalkeeper_ready_successor_contract(
        save_policy_hash=_HASH,
        recovery_policy_hash=_HASH,
        body_hash=_HASH,
    )
    passing = {
        "pelvis_height_m": 0.76,
        "upright_projection": 0.96,
        "root_linear_speed_mps": 0.10,
        "root_angular_speed_rad_s": 0.20,
        "bilateral_support": 1.0,
        "facing_field_cos": 0.96,
        "inside_keeper_region": 1.0,
        "hand_ready_error_rad": 0.08,
        "lateral_acceleration_capacity_mps2": 0.75,
    }
    tracker = SuccessorStateTracker(contract)
    for step in range(49):
        result = tracker.update(SuccessorStateSample(step=step, values=passing))
        assert not result.achieved
    broken = dict(passing)
    broken["root_linear_speed_mps"] = 0.30
    result = tracker.update(SuccessorStateSample(step=49, values=broken))
    assert not result.achieved
    for step in range(50, 100):
        result = tracker.update(SuccessorStateSample(step=step, values=passing))
    assert result.achieved
    assert result.entry_step == 50
    assert result.achieved_step == 99
    assert result.transition_time_s == pytest.approx(2.0)


def test_controlled_fall_exploration_never_grants_promotion() -> None:
    exploration, promotion = build_controlled_fall_safety_profiles()
    controlled = GrowthSafetyObservation(
        phase="GET_UP",
        finite_state=True,
        joint_limit_excess_rad=0.01,
        normalized_actuator_command=0.9,
        head_impact_speed_mps=0.0,
        root_angular_speed_rad_s=2.0,
        self_penetration_m=0.002,
        contacts=("left_hand", "left_lateral_thigh"),
    )
    exploration_decision = evaluate_growth_safety(exploration, controlled)
    assert exploration_decision.passed
    assert not exploration_decision.promotion_eligible
    promotion_decision = evaluate_growth_safety(promotion, controlled)
    assert not promotion_decision.passed
    head_contact = GrowthSafetyObservation(
        phase="ABSORB",
        finite_state=True,
        joint_limit_excess_rad=0.0,
        normalized_actuator_command=0.5,
        head_impact_speed_mps=0.0,
        root_angular_speed_rad_s=1.0,
        self_penetration_m=0.0,
        contacts=("head",),
    )
    assert not evaluate_growth_safety(exploration, head_contact).passed
    assert not evaluate_growth_safety(promotion, head_contact).passed
