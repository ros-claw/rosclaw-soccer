import numpy as np
import pytest

from rosclaw_soccer.rsi.native_reference_authority import (
    reference_contact_target,
    worsens_target_range,
)


def test_reference_authority_is_zero_outside_measured_contact_domain():
    base = np.arange(29, dtype=np.float32) / 100
    reference = base + 10
    for sense in (np.zeros(9), np.r_[0.8, np.zeros(8)]):
        actual, gate = reference_contact_target(base, reference, sense, gain=0.5)
        assert gate == 0
        assert np.array_equal(actual, base)
    sense = np.r_[0.2, np.zeros(8)]
    zero, _ = reference_contact_target(base, reference, sense, gain=0.0)
    assert np.array_equal(zero, base)
    actual, gate = reference_contact_target(base, reference, sense, gain=0.5)
    assert gate == 1
    np.testing.assert_allclose(actual - base, 0.25, atol=3e-8)
    with pytest.raises(ValueError):
        reference_contact_target(base, reference, sense, gain=1.0)


def test_original_pd_range_excess_not_repaired_or_increased():
    bounds = np.tile([-1.0, 1.0], (29, 1))
    base = np.zeros(29)
    base[0] = 1.2
    assert not worsens_target_range(base, base.copy(), bounds)
    candidate = base.copy()
    candidate[0] = 1.1
    assert not worsens_target_range(base, candidate, bounds)
    candidate[0] = 1.3
    assert worsens_target_range(base, candidate, bounds)
    candidate[0], candidate[1] = 1.1, -1.01
    assert worsens_target_range(base, candidate, bounds)
