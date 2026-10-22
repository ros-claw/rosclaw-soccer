"""Two explicit assisted TRAIN recovery courses; no arbitrary failed-case search."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorRecoveryCourse:
    contexts: tuple[int, ...]
    checkpoint_kind: str
    worlds: int
    audit_status: str


def motor_recovery_course(name: str, mode: str) -> MotorRecoveryCourse:
    if name == "legacy4":
        return MotorRecoveryCourse(
            (4, 5, 20, 21), "supervised", 24, "DUAL_MOTOR_ASSISTED24_OWN_NATIVE_AUDIT_PASS"
        )
    if name == "continual-failed3" and mode == "measured_contact_observed_shin":
        return MotorRecoveryCourse(
            (4, 20, 21),
            "ar1-actor-critic",
            18,
            "DUAL_MOTOR_ASSISTED18_CONTINUAL_OWN_NATIVE_AUDIT_PASS",
        )
    raise ValueError("explicit bounded teacher recovery course and mode required")
