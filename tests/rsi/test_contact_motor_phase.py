import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import contact_motor_primitive as basis
from rosclaw_soccer.rsi.contact_motor_contract import motor_delta, validate_policy
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.sim.contracts import hash_json


def test_initial_phase_is_exact_old_arithmetic_and_duration_can_be_learned():
    knots = np.full((3, 12), 0.08)
    limits = np.tile([-1.0, 1.0], (12, 1))
    zero = np.zeros(12)
    old = basis.make_policy(knots, hash_json({"old": True}))
    clone = make_policy(knots, 0.25, hash_json({"new": True}))
    longer = make_policy(knots, 0.1, hash_json({"new": True}))
    for gap in (1.5, 1.1, 0.8, 0.3, 0.2):
        assert np.array_equal(
            motor_delta(old, knots, gap, zero, limits, zero, zero, None),
            motor_delta(clone, knots, gap, zero, limits, zero, zero, None),
        )
    assert np.all(motor_delta(longer, knots, 0.2, zero, limits, zero, zero, None) > 1e-6)


@pytest.mark.parametrize("end", [float("nan"), -0.36, 0.26, True])
def test_invalid_learned_duration_is_not_accepted(end):
    good = make_policy(np.zeros((3, 12)), 0.25, hash_json({"test": True}))
    bad = copy.deepcopy(good)
    bad["phase_gap_end_m"] = end
    if np.isfinite(end):
        bad["policy_hash"] = hash_json({k: v for k, v in bad.items() if k != "policy_hash"})
    with pytest.raises(ValueError):
        validate_policy(bad)
