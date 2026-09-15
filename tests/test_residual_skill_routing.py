import pytest

from rosclaw_soccer.growth.residual_skill_routing import (
    residual_skill_preempted,
    residual_skill_selected,
)


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("option", [False, True])
def test_opt_in_finisher_learning_is_only_inside_admitted_option(teacher, option):
    kwargs = dict(role="finisher", option_only_roles=("finisher",), is_motor_option=option)
    assert residual_skill_selected(**kwargs, is_contact_teacher=teacher) is option
    assert residual_skill_preempted(**kwargs) is (not option)


@pytest.mark.parametrize("role,scope", [("goalkeeper", ("finisher",)), ("finisher", None)])
@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("option", [False, True])
def test_other_roles_and_disabled_configuration_keep_original_slot(role, scope, teacher, option):
    kwargs = dict(role=role, option_only_roles=scope, is_motor_option=option)
    assert residual_skill_selected(**kwargs, is_contact_teacher=teacher) is (teacher and not option)
    assert residual_skill_preempted(**kwargs) is option


def test_option_scope_requires_explicit_strike_coverage_and_is_hash_bound():
    from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig

    with pytest.raises(ValueError, match="strike coverage"):
        IndependentTeamWorldConfig(option_only_residual_roles=("finisher",))
    base = IndependentTeamWorldConfig(strike_residual_enabled=True)
    scoped = IndependentTeamWorldConfig(
        strike_residual_enabled=True, option_only_residual_roles=("finisher",)
    )
    assert scoped.config_hash != base.config_hash
    for scope in ((), ("unknown",), ("finisher", "finisher"), (True,)):
        with pytest.raises(ValueError, match="canonical role scope"):
            IndependentTeamWorldConfig(
                strike_residual_enabled=True, option_only_residual_roles=scope
            )
