import numpy as np
import pytest
from test_motion_reference_timing import policy

from rosclaw_soccer.rsi.whole_joint_reference import (
    WholeJointReferencePlan,
    condition_whole_joint_reference,
)


def plan(scale=0.03, mapping=None):
    return WholeJointReferencePlan(
        20,
        tuple(range(29)) if mapping is None else mapping,
        ((float(scale),) * 29, (-float(scale),) * 29),
    )


def test_disjoint_blocks_act_on_all29_reference_coordinates_without_larger_authority():
    p = policy()
    q, v, anchors = p.motion_joint_pos.copy(), p.motion_joint_vel.copy(), p.motion_body_pos.copy()
    result = condition_whole_joint_reference(p, plan())
    assert len(result["blocks"]) == 5 and result["applied"]
    np.testing.assert_allclose(p.motion_joint_pos[32] - q[32], np.full(29, 0.03), atol=1e-15)
    np.testing.assert_allclose(p.motion_joint_pos[56] - q[56], np.full(29, -0.03), atol=1e-15)
    for i in (20, 44, 68):
        np.testing.assert_array_equal(p.motion_joint_pos[i], q[i])
        np.testing.assert_array_equal(p.motion_joint_vel[i], v[i])
    np.testing.assert_array_equal(p.motion_body_pos, anchors)
    assert p.time_step == 20 and result["maximum_reference_offset_rad"] <= 0.06
    assert not result["actual_joint_targets_overridden"]


def test_named_permutation_not_assumed_joint_order():
    p = policy()
    mapping = tuple(map(int, np.roll(np.arange(29), 3)))
    coeff = np.zeros((2, 29))
    coeff[0, 4] = 0.02
    before = p.motion_joint_pos.copy()
    condition_whole_joint_reference(
        p, WholeJointReferencePlan(20, mapping, tuple(tuple(r) for r in coeff))
    )
    changed = p.motion_joint_pos[32] - before[32]
    assert np.flatnonzero(changed).tolist() == [mapping[4]]


def test_zero_is_exact_and_rejection_never_partially_mutates_reference():
    p = policy()
    q, v = p.motion_joint_pos, p.motion_joint_vel
    assert not condition_whole_joint_reference(p, plan(0))["applied"]
    assert p.motion_joint_pos is q and p.motion_joint_vel is v
    with pytest.raises(ValueError, match="bounded"):
        condition_whole_joint_reference(p, plan(0.07))
    with pytest.raises(ValueError, match="permutation"):
        condition_whole_joint_reference(p, plan(mapping=(0,) * 29))
    assert p.motion_joint_pos is q and p.motion_joint_vel is v
