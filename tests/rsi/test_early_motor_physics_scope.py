"""No learning/teacher scope expansion in whole-six motor reconstruction."""

import pytest

from scripts.rsi_audit_early_motor_physics import require_scope


def scope():
    return dict(
        profile="six-early-original-motor-physics-v1",
        mode="SIM_ONLY",
        source_cases=6,
        physical_step_budget=18000,
        motor_decisions=1800,
        recurrent_updates=1620,
        new_optimizer_updates=0,
        new_foundation_inferences=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
        independent_search_qualification_claimed=False,
    )


def test_exact_whole_six_motor_and_physics_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_absent_or_changed_admission_rejected(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)
