from dataclasses import replace

import pytest

from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig


def test_opt_in_pass_approach_is_bounded_and_preserves_disabled_identity():
    baseline = IndependentTeamWorldConfig()
    assert baseline.precontact_pass_standoff_m is None
    assert replace(baseline, precontact_pass_standoff_m=0.35).config_hash != baseline.config_hash
    for invalid in (0.0, 0.19, 0.51, float("nan"), float("inf"), True, "0.35"):
        with pytest.raises(ValueError):
            replace(baseline, precontact_pass_standoff_m=invalid)


def test_option_ankle_braking_is_explicit_bounded_and_hash_bound():
    baseline = IndependentTeamWorldConfig()
    assert baseline.option_ankle_roll_braking_damping is None
    enabled = replace(baseline, option_ankle_roll_braking_damping=8.0)
    assert enabled.config_hash != baseline.config_hash
    for invalid in (0.0, 5.9, 20.1, float("nan"), float("inf"), True, "8"):
        with pytest.raises(ValueError):
            replace(baseline, option_ankle_roll_braking_damping=invalid)


def test_option_joint_guard_is_explicit_bounded_and_hash_bound():
    baseline = IndependentTeamWorldConfig()
    assert baseline.option_joint_guard_margin_rad is None
    assert replace(baseline, option_joint_guard_margin_rad=0.08).config_hash != baseline.config_hash
    for invalid in (0.0, 0.039, 0.121, float("nan"), float("inf"), True, "0.08"):
        with pytest.raises(ValueError):
            replace(baseline, option_joint_guard_margin_rad=invalid)
