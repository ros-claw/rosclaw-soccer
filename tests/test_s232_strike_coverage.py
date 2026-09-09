from dataclasses import asdict

import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    TacticalIntent,
    _residual_intent_enabled,
)
from rosclaw_soccer.training.contact_control_profile import ContactControlProfile
from rosclaw_soccer.training.near_ball_learning_audit import _verify_contact_profile


def test_shoot_and_distribution_are_explicit_not_default():
    for intent in (TacticalIntent.SHOOT, TacticalIntent.DISTRIBUTE):
        assert not _residual_intent_enabled(intent, strike_enabled=False)
        assert _residual_intent_enabled(intent, strike_enabled=True)
    assert not _residual_intent_enabled(TacticalIntent.SAVE, strike_enabled=True)
    for intent in (TacticalIntent.PASS, TacticalIntent.CARRY, TacticalIntent.RECEIVE):
        assert _residual_intent_enabled(intent, strike_enabled=False)


@pytest.mark.parametrize("value", [1, "true", None])
def test_coverage_requires_boolean(value):
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(strike_residual_enabled=value)
    with pytest.raises(ValueError):
        ContactControlProfile(strike_residual_enabled=value)
    with pytest.raises(ValueError):
        _residual_intent_enabled(TacticalIntent.SHOOT, strike_enabled=value)


def test_profile_and_audit_bind_actual_strike_coverage():
    profile = ContactControlProfile(strike_residual_enabled=True)
    world, teacher = profile.apply(IndependentTeamWorldConfig(), G1LocomotionContactTeacherConfig())
    assert world.strike_residual_enabled
    assert world.config_hash != IndependentTeamWorldConfig().config_hash
    report = dict(world_config=asdict(world), contact_teacher_config=asdict(teacher))
    _verify_contact_profile(report, profile)
    report["world_config"]["strike_residual_enabled"] = False
    with pytest.raises(ValueError, match="contact control profile"):
        _verify_contact_profile(report, profile)
