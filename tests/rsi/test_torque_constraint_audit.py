import numpy as np
import pytest

from rosclaw_soccer.rsi.torque_constraint_audit import audit_applied_torques


def audit(applied, parent, velocity=None):
    return audit_applied_torques(
        np.array(applied),
        np.array(parent),
        np.zeros_like(applied) if velocity is None else np.array(velocity),
        np.array([100.0]),
        guard_scale=0.70,
        parent_deviation_ratio=0.08,
        delta_ratio=0.08,
        maximum_power_w=100.0,
    )


def test_sequential_clipping_violation_is_detected():
    r = audit([[7.0]], [[20.0]])
    assert r["parent_deviation_violations"] == 1
    assert r["nonparent_rate_violations"] == 0


def test_rate_uses_actual_prior_including_parent_fallback():
    r = audit([[80.0], [70.0]], [[80.0], [70.1]])
    assert r["nonparent_rate_violations"] == 1
    assert r["parent_valued_steps"] == 1
    assert r["total_amplitude_violations"] == 0


def test_parent_power_exception_not_certified_as_candidate():
    r = audit([[80.0], [79.0]], [[80.0], [80.0]], [[10.0], [10.0]])
    assert r["nonparent_power_violations"] == 1
    assert r["maximum_parent_valued_power_w"] == 800.0
    assert r["nonparent_amplitude_violations"] == 1


def test_valid_commands_pass():
    r = audit([[5.0], [7.0]], [[0.0], [0.0]])
    assert all(value == 0 for key, value in r.items() if key.endswith("violations"))


def test_nan_is_rejected_not_counted_as_pass():
    with pytest.raises(ValueError):
        audit([[np.nan]], [[0.0]])
