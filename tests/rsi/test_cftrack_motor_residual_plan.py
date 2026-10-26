from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_motor_residual_plan import CFtrackMotorResidualPlan


def plan():
    delta = [0.0] * 29
    delta[8], delta[12] = -0.15, 0.1
    return CFtrackMotorResidualPlan(0, 46, 24, tuple(delta), "sha256:" + "a" * 64)


def test_c2_motor_authority_is_distinct_from_reference_and_zero_branch_exact():
    p = plan()
    pos, vel = p.offsets()
    assert pos.shape == vel.shape == (49, 29)
    np.testing.assert_array_equal(pos[[0, 48]], np.zeros((2, 29)))
    np.testing.assert_array_equal(vel[[0, 24, 48]], np.zeros((3, 29)))
    np.testing.assert_array_equal(pos[24], p.delta_hg)
    limits = np.tile([-1.0, 1.0], (29, 1))
    original = np.zeros(29)
    original[0] = 2  # Unchanged original coordinate is not silently clipped.
    original[4] = -0.0
    for tick in [0, 45, 46, 94, 95, 299]:
        np.testing.assert_array_equal(p.apply(original, counter=tick, limits_hg=limits), original)
        np.testing.assert_array_equal(
            p.apply(original, counter=tick, limits_hg=limits).view(np.uint64),
            original.view(np.uint64),
        )
    actual = p.apply(original, counter=70, limits_hg=limits)
    np.testing.assert_allclose(actual - original, p.delta_hg)
    assert original[8] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(start_control=-1),
        dict(origin_counter=True),
        dict(origin_counter=50),
        dict(peak_control=0),
        dict(peak_control=1),
        dict(delta_hg=(0.251,) * 29),
        dict(delta_hg=(float("nan"),) * 29),
        dict(delta_hg=(0.0,) * 28),
        dict(donor_policy_hash="unbound"),
    ],
)
def test_invalid_or_too_fast_motor_authority_is_rejected(change):
    with pytest.raises(ValueError):
        replace(plan(), **change).offsets()


def test_limit_violation_rejects_without_modifying_caller_target():
    p = plan()
    original = np.zeros(29)
    original[8] = -0.95
    before = original.copy()
    with pytest.raises(ValueError, match="no clipping"):
        p.apply(original, counter=70, limits_hg=np.tile([-1.0, 1.0], (29, 1)))
    np.testing.assert_array_equal(original, before)


def test_installation_requires_exact_donor_hash_and_control_boundary():
    p = plan()
    p.require_donor(policy_hash=p.donor_policy_hash, counter=0)
    for value, tick in [
        ("sha256:" + "b" * 64, 0),
        (p.donor_policy_hash, 1),
        (p.donor_policy_hash, False),
    ]:
        with pytest.raises(ValueError, match="donor"):
            p.require_donor(policy_hash=value, counter=tick)
