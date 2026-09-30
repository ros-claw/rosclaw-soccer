import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_motor_primitive import (
    CAP_RAD,
    SLEW_RAD,
    make_policy,
    motor_delta,
    validate_policy,
)
from rosclaw_soccer.sim.contracts import hash_json


def test_sealed_policy_rejects_tamper_and_hardware_claim():
    policy = make_policy(np.zeros((3, 12)), hash_json({"experiment": 303}))
    validate_policy(policy)
    for key, value in (("promotion_authorized", True), ("activation_ceiling", "REAL")):
        bad = copy.deepcopy(policy)
        bad[key] = value
        bad["policy_hash"] = hash_json({k: v for k, v in bad.items() if k != "policy_hash"})
        with pytest.raises(ValueError):
            validate_policy(bad)
    policy["knots_rad"][0][0] = 0.1
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_causal_phase_bounded_bilateral_and_smooth_release():
    knots = np.full((3, 12), CAP_RAD)
    baseline = np.zeros(12)
    limits = np.tile([-1.0, 1.0], (12, 1))
    zero = np.zeros(12)
    assert np.array_equal(motor_delta(knots, 1.5, baseline, limits, zero, zero, None), zero)
    delta = zero
    for _ in range(20):
        next_delta = motor_delta(knots, 0.825, baseline, limits, delta, zero, None)
        assert np.max(np.abs(next_delta - delta)) <= SLEW_RAD + 1e-12
        delta = next_delta
    assert np.allclose(delta, CAP_RAD)
    contact = delta.copy()
    for frame in range(1, 25):
        next_delta = motor_delta(knots, 0.825, baseline, limits, delta, contact, frame)
        assert np.max(np.abs(next_delta - delta)) <= SLEW_RAD + 1e-12
        delta = next_delta
    assert np.allclose(delta, 0)


def test_existing_limit_violation_is_not_amplified_and_nan_rejected():
    knots = np.full((3, 12), CAP_RAD)
    baseline = np.full(12, 1.1)
    limits = np.tile([-1.0, 1.0], (12, 1))
    zero = np.zeros(12)
    assert np.array_equal(motor_delta(knots, 0.825, baseline, limits, zero, zero, None), zero)
    with pytest.raises(ValueError):
        motor_delta(knots, float("nan"), baseline, limits, zero, zero, None)
