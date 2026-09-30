import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import contact_motor_primitive as preparation
from rosclaw_soccer.rsi import contact_motor_strike as strike
from rosclaw_soccer.rsi.contact_motor_contract import motor_delta, validate_policy
from rosclaw_soccer.sim.contracts import hash_json


def test_new_profile_keeps_control_at_measured_contact_gap_without_reinterpreting_old():
    knots = np.full((3, 12), 0.08)
    commitment = hash_json({"experiment": 304})
    old = preparation.make_policy(knots, commitment)
    new = strike.make_policy(knots, commitment)
    zero = np.zeros(12)
    limits = np.tile([-1, 1], (12, 1))
    assert old["policy_hash"] != new["policy_hash"]
    assert np.allclose(
        motor_delta(old, knots, 0.2, zero, limits, zero, zero, None), zero, atol=1e-16, rtol=0
    )
    delta = motor_delta(new, knots, 0.2, zero, limits, zero, zero, None)
    assert np.all(delta > 0)
    assert np.max(delta) <= preparation.SLEW_RAD
    for policy in (old, new):
        validate_policy(policy)
        assert np.array_equal(motor_delta(policy, knots, 1.5, zero, limits, zero, zero, None), zero)
        release = motor_delta(policy, knots, 0.2, zero, limits, delta, delta, 1)
        assert np.max(release) < np.max(delta)


def test_strike_phase_and_source_are_sealed_even_when_outer_hash_is_recomputed():
    new = strike.make_policy(np.zeros((3, 12)), hash_json({"experiment": 304}))
    for key, value in (
        ("phase_gap_end_m", 0.25),
        ("basis_source_hash", "sha256:fake"),
        ("contract_source_hash", "sha256:fake"),
        ("activation_ceiling", "REAL"),
    ):
        bad = copy.deepcopy(new)
        bad[key] = value
        bad["policy_hash"] = hash_json({k: v for k, v in bad.items() if k != "policy_hash"})
        with pytest.raises(ValueError):
            validate_policy(bad)
