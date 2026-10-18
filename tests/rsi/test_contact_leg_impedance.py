import numpy as np
import pytest

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.contact_leg_impedance import half_contact_leg_stiffness
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES


@pytest.mark.parametrize("side", [0, 1, None])
def test_only_causal_contact_leg_stiffness_is_changed_without_mutating_parent(side):
    kp = np.arange(1, 30, dtype=np.float64)
    original = kp.copy()
    candidate = half_contact_leg_stiffness(kp, side)
    indices = [] if side is None else [G1_DDS_JOINT_NAMES.index(n) for n in LEG_NAMES[side]]
    expected = original.copy()
    expected[indices] *= 0.5
    assert np.array_equal(candidate, expected)
    assert kp.tobytes() == original.tobytes()
    assert not np.shares_memory(candidate, kp)


@pytest.mark.parametrize("side", [True, -1, 2, "left"])
def test_bad_side_is_not_motion_authority(side):
    with pytest.raises(ValueError):
        half_contact_leg_stiffness(np.ones(29), side)
