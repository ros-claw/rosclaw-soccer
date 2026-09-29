"""The richer proprioceptive policy remains zero-authority and fail-closed."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    FEATURE_COUNT,
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
    ReceivingProtectedKinematicExpert,
)
from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor() -> ReceivingKinematicTemporalExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingKinematicTemporalExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
    )


def _observation() -> SimpleNamespace:
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[2] = 0.7
    qpos[36:39] = [0.4, 0.14, 0.1]
    qvel[35] = -1.25
    return SimpleNamespace(
        qpos=tuple(qpos),
        qvel=tuple(qvel),
        frame=20,
        last_own_foot_contact_time_sec=None,
        foot_kinematics=SimpleNamespace(
            foot_position_world_m=((0.1, 0.1, 0.05), (0.2, 0.1, 0.05)),
            foot_linear_velocity_world_mps=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
        shin_clearance=SimpleNamespace(clearance_m=(0.04, 0.06)),
    )


def test_finite_measured_48d_features_and_required_sensors() -> None:
    observation = _observation()
    features = ReceivingKinematicTemporalExpert.features(observation)
    assert len(features) == FEATURE_COUNT
    assert features[22:24] == pytest.approx((0.4, 0.6))
    observation.shin_clearance = None
    with pytest.raises(ValueError, match="same-frame measured"):
        ReceivingKinematicTemporalExpert.features(observation)


def test_zero_kinematic_policy_keeps_parent_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    actor = _actor()
    assert actor.propose(_observation()) == (0.0,) * 29
    assert actor.requires_shin_clearance
    assert actor.activation_ceiling == "SIM_ONLY"
    assert len(actor.observed_features[0]) == FEATURE_COUNT


def test_kinematic_weights_reject_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite bounded kinematic"):
        KinematicMotorWeights(output_bias=(float("nan"),) + (0.0,) * 11)


def test_legacy_actor_embedding_keeps_new_sensors_initially_inert() -> None:
    legacy = TemporalMotorWeights(input_matrix=(0.2,) * 320, output_bias=(0.1,) * 12)
    embedded = KinematicMotorWeights.from_legacy(legacy)
    assert embedded.input_matrix[:10] == legacy.input_matrix[:10]
    assert all(value == 0.0 for value in embedded.input_matrix[10:48])
    assert embedded.output_bias == legacy.output_bias


def test_embedded_actor_logits_are_bitwise_legacy_equal() -> None:
    rng = np.random.default_rng(42)
    legacy = TemporalMotorWeights(
        input_matrix=tuple(float(value) for value in rng.normal(0, 0.1, 320)),
        input_bias=tuple(float(value) for value in rng.normal(0, 0.1, 32)),
        output_matrix=tuple(float(value) for value in rng.normal(0, 0.1, 384)),
        output_bias=tuple(float(value) for value in rng.normal(0, 0.1, 12)),
    )
    actor = _actor()
    actor.policy = legacy
    actor.kinematic_policy = KinematicMotorWeights.from_legacy(legacy)
    features = np.asarray(ReceivingKinematicTemporalExpert.features(_observation()))
    assert np.array_equal(
        actor._motor_logits(features),
        actor.__class__.__mro__[1]._motor_logits(actor, features[:10]),
    )


def test_old_skill_protection_uses_measured_initial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    measured = ReceivingTemporalMotorExpert.features(_observation())
    actor = ReceivingProtectedKinematicExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        protected_initial_features=(measured, measured),
    )
    assert actor.propose(_observation()) == (0.0,) * 29
    assert actor.protected_episode is True
