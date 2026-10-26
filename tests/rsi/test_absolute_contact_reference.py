from types import SimpleNamespace

import numpy as np
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
    assert policy.motion_joint_pos[24, 28] == np.float32(0.1)
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
    policy.motion_joint_pos = np.full((200, 29), 0.5, np.float32)
    policy.motion_joint_vel = np.zeros((200, 29), np.float32)
    plan = AbsoluteContactReferencePlan(
        int(policy.time_step), tuple(range(29)), (0.6,) + (0.5,) * 28, 14
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
