import pytest

from rosclaw_soccer.rsi.motor_recovery_course import motor_recovery_course


def test_legacy_supervised_course_not_silently_changed():
    c = motor_recovery_course("legacy4", "reference")
    assert c.contexts == (4, 5, 20, 21) and c.worlds == 24
    assert c.checkpoint_kind == "supervised"


def test_continual_failed_context_course_has_own_budget_and_audit_identity():
    c = motor_recovery_course("continual-failed3", "measured_contact_observed_shin")
    assert c.contexts == (4, 20, 21) and c.worlds == 18
    assert c.checkpoint_kind == "ar1-actor-critic"
    assert "18_CONTINUAL" in c.audit_status


def test_predictive_teacher_is_a_separate_profile_without_more_authority_or_worlds():
    c = motor_recovery_course("continual-predictive3", "measured_contact_observed_shin")
    assert c.contexts == (4, 20, 21) and c.worlds == 18
    assert c.clearance_profile == "anticipatory-priority-v1"
    assert "18_PREDICTIVE" in c.audit_status


@pytest.mark.parametrize(
    "course,mode", [("arbitrary", "reference"), ("continual-failed3", "reference")]
)
def test_unreviewed_teacher_search_not_authorized(course, mode):
    with pytest.raises(ValueError):
        motor_recovery_course(course, mode)
