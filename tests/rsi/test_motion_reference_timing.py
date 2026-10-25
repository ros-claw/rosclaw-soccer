from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_reference_timing import MotionTimingPlan, condition_motion_reference


def policy():
    t = np.arange(100)[:, None] * 0.02
    return SimpleNamespace(
        runtime_mode="sim",
        use_body_frame_ball=False,
        time_step=20,
        WARMUP_STEPS=0,
        control_dt=0.02,
        motion_total_steps=100,
        motion_joint_pos=np.repeat(t**2, 29, axis=1),
        motion_joint_vel=np.repeat(2 * t, 29, axis=1),
        motion_body_pos=np.repeat(t[:, None], 6, axis=1),
        motion_body_quat=np.tile([1.0, 0.0, 0.0, 0.0], (100, 2, 1)),
    )


def test_zero_plan_preserves_array_identity_and_all_software():
    p = policy()
    before = vars(p).copy()
    r = condition_motion_reference(p, MotionTimingPlan(20, 24, 0.0))
    assert not r["applied"] and r["zero_branch_exact"]
    assert all(value is vars(p)[key] for key, value in before.items())


@pytest.mark.parametrize("shift", [-3.0, 3.0])
def test_hermite_and_chain_rule_correct_on_quadratic_reference(shift):
    p = policy()
    # body position must have proper XYZ dimension; two bodies.
    p.motion_body_pos = np.tile(np.arange(100)[:, None, None] * 0.02, (1, 2, 3))
    original = {k: v.copy() for k, v in vars(p).items() if isinstance(v, np.ndarray)}
    condition_motion_reference(p, MotionTimingPlan(20, 24, shift))
    rows = np.arange(21, 44)
    u = (rows - 20) / 24
    sampled = (rows + shift * np.sin(np.pi * u) ** 2) * 0.02
    rate = 1 + shift * np.pi / 24 * np.sin(2 * np.pi * u)
    np.testing.assert_allclose(p.motion_joint_pos[rows, 0], sampled**2, atol=1e-15)
    np.testing.assert_allclose(p.motion_joint_vel[rows, 0], 2 * sampled * rate, atol=1e-13)
    for name, old in original.items():
        np.testing.assert_array_equal(getattr(p, name)[:21], old[:21])
        np.testing.assert_array_equal(getattr(p, name)[44:], old[44:])
    assert p.time_step == 20


def test_shortest_arc_quaternion_and_rejection_is_atomic():
    p = policy()
    p.motion_body_pos = np.zeros((100, 2, 3))
    p.motion_body_quat[22] *= -1
    condition_motion_reference(p, MotionTimingPlan(20, 24, 3.0))
    np.testing.assert_allclose(abs(p.motion_body_quat[:, :, 0]), 1)
    np.testing.assert_allclose(p.motion_body_quat[:, :, 1:], 0)
    before = p.motion_joint_pos
    with pytest.raises(ValueError, match="future complete"):
        condition_motion_reference(p, MotionTimingPlan(19, 24, 3.0))
    assert p.motion_joint_pos is before


@pytest.mark.parametrize(
    "plan",
    [
        MotionTimingPlan(20, 12, 1.0),
        MotionTimingPlan(20, 24, 4.0),
        MotionTimingPlan(20, 24, float("nan")),
    ],
)
def test_invalid_plans_rejected(plan):
    with pytest.raises(ValueError, match="bounded"):
        condition_motion_reference(policy(), plan)
