import pytest

from rosclaw_soccer.growth.residual_skill_routing import prospective_contact_preempted
from rosclaw_soccer.growth.role_self_model import TacticalIntent


@pytest.mark.parametrize("prospective", [False, True])
@pytest.mark.parametrize("pass_enabled", [False, True])
@pytest.mark.parametrize("intent", list(TacticalIntent))
def test_motor_can_only_preempt_its_enabled_contact_skill(prospective, pass_enabled, intent):
    assert prospective_contact_preempted(
        prospective_enabled=prospective,
        pass_enabled=pass_enabled,
        intent=intent,
    ) == (
        prospective
        and (intent is TacticalIntent.SHOOT or pass_enabled and intent is TacticalIntent.PASS)
    )


def test_shoot_only_prospective_motor_cannot_disable_outlet_pass_teacher():
    assert not prospective_contact_preempted(
        prospective_enabled=True, pass_enabled=False, intent="pass"
    )
    assert prospective_contact_preempted(
        prospective_enabled=True, pass_enabled=False, intent="shoot"
    )


@pytest.mark.parametrize("bad", [1, None, "true"])
def test_preemption_contract_is_explicit(bad):
    with pytest.raises(ValueError, match="explicit"):
        prospective_contact_preempted(prospective_enabled=bad, pass_enabled=False, intent="pass")
