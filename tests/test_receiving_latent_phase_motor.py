"""Latent exploration remains constant over the measured post-contact phase."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_latent_phase_motor import ReceivingLatentPhaseMotor
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(std: float, seed: int = 7) -> ReceivingLatentPhaseMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingLatentPhaseMotor(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        references=(ref,),
        corrected_experts=("high",),
        foot_gain=0.3,
        low_post_weights=(0.0,) * 12,
        protected_initial_features=((0.1,) * 10,) * 6,
        latent_std=std,
        latent_seed=seed,
    )


def test_latent_is_fixed_seeded_bounded_and_zero_equivalent() -> None:
    first = _actor(0.4)
    replay = _actor(0.4)
    assert first.latent_offset == replay.latent_offset
    assert max(abs(value) for value in first.latent_offset) <= 1.2
    x = (0.0,) * 48
    assert np.array_equal(first._post_logits(x), first._post_logits(x))
    assert np.array_equal(first._post_logits(x), replay._post_logits(x))
    assert not np.any(_actor(0.0)._post_logits(x))


def test_per_frame_noise_cannot_be_combined_with_latent() -> None:
    actor = _actor(0.2)
    actor.post_exploration_std = 0.12
    with pytest.raises(ValueError, match="episode-coherent"):
        actor.__post_init__()
