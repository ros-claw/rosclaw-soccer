import pytest

from rosclaw_soccer.rsi.paid_motor_schedule import lifecycle_motor_schedule
from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_standing_lifecycle


def test_default_old_modes_and_explicit_third_ancestor_are_distinct():
    assert lifecycle_motor_schedule(0, 3) == lifecycle_motor_schedule(1, 3) == ("base",) * 3
    assert lifecycle_motor_schedule(2, 3) == ("base", "student", "student")
    teacher = lifecycle_motor_schedule(2, 3, ("base", "student", "base"))
    assert teacher[:2] == lifecycle_motor_schedule(2, 3)[:2] and teacher[2] == "base"


def test_first_enabled_requires_explicit_commitment_and_does_not_change_legacy_default():
    assert lifecycle_motor_schedule(2, 3, ("student",) * 3) == ("student",) * 3
    assert lifecycle_motor_schedule(2, 3)[0] == "base"


def test_first_enabled_world_cannot_be_audited_as_unchanged_first_motor():
    with pytest.raises(ValueError, match="caller commitment"):
        audit_standing_lifecycle(
            None,
            None,
            None,
            dict(motor_schedule=["student"] * 3),
            {},
            {},
            {},
            variant=2,
            skills=3,
            count_inference=lambda: None,
            motor_schedule=("base", "student", "student"),
        )


@pytest.mark.parametrize(
    "variant,schedule",
    [
        (1, ("base",) * 3),
        (2, ["base"] * 3),
        (2, ("base",) * 2),
        (2, ("base", "teacher", "base")),
        (False, ("base",) * 3),
    ],
)
def test_uncommitted_or_invalid_schedule_cannot_allocate_physics(variant, schedule):
    with pytest.raises(ValueError):
        run_standing_lifecycle(
            None, {}, None, variant=variant, skills=3, store=lambda *_: {}, motor_schedule=schedule
        )


@pytest.mark.parametrize("commitment", [None, ("base", "student", "student")])
def test_audit_does_not_accept_self_declared_schedule_as_student_evidence(commitment):
    # Must fail BEFORE touching any auditor/model. Legacy callers cannot be
    # tricked into verifying R2 work as a student by changing a world field.
    with pytest.raises(ValueError, match="caller commitment"):
        audit_standing_lifecycle(
            None,
            None,
            None,
            dict(motor_schedule=["base", "student", "base"]),
            {},
            {},
            {},
            variant=2,
            skills=3,
            count_inference=lambda: None,
            motor_schedule=commitment,
        )
