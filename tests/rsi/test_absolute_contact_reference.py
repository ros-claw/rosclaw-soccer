from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.absolute_contact_reference import (
    AbsoluteContactReferencePlan,
    condition_absolute_contact_reference,
)
from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash


def test_absolute_waypoint_resolves_against_reference_not_another_body_pose():
    mapping = tuple(reversed(range(29)))
    q = np.full((100, 29), 0.5, np.float32)
    policy = SimpleNamespace(
        motion_joint_pos=q,
        motion_joint_vel=np.zeros_like(q),
        motion_total_steps=100,
        time_step=10,
        WARMUP_STEPS=0,
        control_dt=0.02,
        runtime_mode="sim",
        use_body_frame_ball=False,
    )
    target = (0.1,) + (0.5,) * 28
    plan = AbsoluteContactReferencePlan(10, mapping, target, 14)
    limits = np.tile([-2.0, 2.0], (29, 1))
    receipt = condition_absolute_contact_reference(policy, plan, limits_hg=limits)
    assert receipt["applied"]
    assert abs(float(policy.motion_joint_pos[24, 28]) - 0.1) <= 1e-6
    assert receipt["encoded_waypoint_error_rad"] <= 1e-6
    np.testing.assert_array_equal(policy.motion_joint_pos[:11], q[:11])
    np.testing.assert_array_equal(policy.motion_joint_pos[58:], q[58:])
    assert receipt["maximum_reference_offset_rad"] > 0.35


def test_absolute_native_future_matches_actual_neural_targets(tmp_path):
    p, f = setup(tmp_path)
    p.world.advance({p.prefix: p.pending.copy()})
    p.commit()
    policy = p.policy
    policy.control_dt = 0.02
    policy.motion_total_steps = 200
    policy.motion_joint_pos = np.zeros((200, 29), np.float32)
    policy.motion_joint_vel = np.zeros((200, 29), np.float32)
    plan = AbsoluteContactReferencePlan(
        int(policy.time_step), tuple(range(29)), (0.03,) + (0.0,) * 28, 14
    )
    before = policy_state_hash(policy)
    predicted = f.forecast(controls=4, before_proposal=True, reference_plan=plan)
    assert predicted["reference_conditioned"] and policy_state_hash(policy) == before
    assert condition_absolute_contact_reference(
        policy, plan, limits_hg=p.world.model.jnt_range[p.layout.joints].astype(np.float64)
    )["applied"]
    actual = []
    for _ in range(4):
        target, _ = p.propose()
        actual.append(p.world.advance({p.prefix: target}))
        p.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in actual]))


def test_bounded_curve_avoids_return_segment_overshoot_without_clipping():
    mapping = tuple(reversed(range(29)))
    q = np.zeros((100, 29), np.float64)
    q[30:36, 28] = 0.9
    policy = SimpleNamespace(
        motion_joint_pos=q,
        motion_joint_vel=np.zeros_like(q),
        motion_total_steps=100,
        time_step=10,
        WARMUP_STEPS=0,
        control_dt=0.02,
        runtime_mode="sim",
        use_body_frame_ball=False,
    )
    limits = np.tile([-1.0, 1.0], (29, 1))
    target = (0.3,) + (0.0,) * 28
    original = (policy.motion_joint_pos.copy(), policy.motion_joint_vel.copy())
    fixed = AbsoluteContactReferencePlan(10, mapping, target, 14)
    assert not condition_absolute_contact_reference(policy, fixed, limits_hg=limits)["applied"]
    np.testing.assert_array_equal(policy.motion_joint_pos, original[0])
    np.testing.assert_array_equal(policy.motion_joint_vel, original[1])
    bounded = AbsoluteContactReferencePlan(10, mapping, target, 14, bounded_curve=True)
    pos, vel = bounded.offsets(policy, limits_hg=limits)
    assert abs(vel).max() <= 6.0 + 1e-8
    assert abs(pos[14, 28] - 0.3) < 1e-8
    assert vel[14, 28] == pytest.approx(0.0, abs=1e-8)
    receipt = condition_absolute_contact_reference(policy, bounded, limits_hg=limits)
    assert receipt["applied"]
    assert abs(policy.motion_joint_pos).max() <= 1.0
    np.testing.assert_array_equal(policy.motion_joint_pos[:11], q[:11])
    np.testing.assert_array_equal(policy.motion_joint_pos[58:], q[58:])
    np.testing.assert_array_equal(policy.motion_joint_pos[:, :28], q[:, :28])


def test_bounded_curve_impossible_waypoint_is_atomic_abstention():
    from rosclaw_soccer.rsi.bounded_contact_reference_curve import bounded_contact_curve

    q = np.zeros((49, 29), np.float64)
    target = np.zeros(29, np.float64)
    target[0] = 2.0
    with pytest.raises(ValueError, match="infeasible"):
        bounded_contact_curve(q, target, np.tile([-1.0, 1.0], (29, 1)), 14)
    np.testing.assert_array_equal(q, np.zeros_like(q))


def test_bounded_curve_continuous_velocity_and_c2_boundary_certificate():
    from scipy.interpolate import BSpline

    from rosclaw_soccer.rsi.bounded_contact_reference_curve import bounded_contact_curve

    q, target = np.zeros((49, 29)), np.zeros(29)
    target[:12] = np.linspace(-0.4, 0.4, 12)
    pos, vel, certificate = bounded_contact_curve(q, target, np.tile([-1.0, 1.0], (29, 1)), 14)
    spline = BSpline(certificate["knots_seconds"], np.array(certificate["coefficients_hg12"]).T, 5)
    np.testing.assert_allclose(spline(np.arange(49) * 0.02), pos[:, :12], atol=1e-8)
    np.testing.assert_allclose(spline.derivative()(np.arange(49) * 0.02), vel[:, :12], atol=1e-8)
    assert abs(spline.derivative()(np.linspace(0, 0.96, 10001))).max() <= 6.0 + 1e-8
    for order in (0, 1, 2):
        np.testing.assert_allclose(spline.derivative(order)([0.0, 0.96]), 0.0, atol=1e-8)
    # Triple contact knot of a quintic is C2, not an acceleration jump.
    np.testing.assert_allclose(
        spline.derivative(2)(0.28 - 1e-10), spline.derivative(2)(0.28 + 1e-10), atol=1e-5
    )


def test_bounded_native_future_matches_actual_neural_targets(tmp_path):
    p, f = setup(tmp_path)
    p.world.advance({p.prefix: p.pending.copy()})
    p.commit()
    policy = p.policy
    policy.control_dt = 0.02
    policy.motion_total_steps = 200
    policy.motion_joint_pos = np.zeros((200, 29), np.float32)
    policy.motion_joint_vel = np.zeros((200, 29), np.float32)
    plan = AbsoluteContactReferencePlan(
        int(policy.time_step), tuple(range(29)), (0.03,) + (0.0,) * 28, 14, bounded_curve=True
    )
    before = policy_state_hash(policy)
    predicted = f.forecast(controls=4, before_proposal=True, reference_plan=plan)
    assert predicted["reference_conditioned"] and policy_state_hash(policy) == before
    assert condition_absolute_contact_reference(
        policy, plan, limits_hg=p.world.model.jnt_range[p.layout.joints].astype(np.float64)
    )["applied"]
    actual = []
    for _ in range(4):
        target, _ = p.propose()
        actual.append(p.world.advance({p.prefix: target}))
        p.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in actual]))
