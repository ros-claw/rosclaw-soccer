"""Neural online residual remains zero-authority until trained and qualified."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_composed_neural_residual import ReceivingComposedNeuralResidual
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(weights: KinematicMotorWeights) -> ReceivingComposedNeuralResidual:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    reference = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingComposedNeuralResidual(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        references=(reference,),
        corrected_experts=("high",),
        foot_gain=0.3,
        low_post_weights=(0.0,) * 12,
        neural_weights=weights,
    )


def test_zero_neural_policy_keeps_composed_parent_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(KinematicMotorWeights())
    assert actor.propose(SimpleNamespace(frame=25)) == (0.0,) * 29
    assert actor.observed_frames == [25]
    assert actor.sampled_logits == [(0.0,) * 12]


def test_parent_fallback_blocks_even_nonzero_neural_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "parent"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    actor = _actor(KinematicMotorWeights(output_bias=(0.2,) * 12))
    assert actor.propose(SimpleNamespace(frame=25)) == (0.0,) * 29
    assert actor.observed_frames == []


def test_neural_residual_stays_bounded_and_same_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "low"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(KinematicMotorWeights(output_bias=(0.2,) * 12))
    target = actor.propose(SimpleNamespace(frame=25))
    np.testing.assert_allclose(target[:12], 0.15 * np.tanh(0.2))
    assert target[12:] == (0.0,) * 17
    assert len(actor.observed_features[0]) == 48
