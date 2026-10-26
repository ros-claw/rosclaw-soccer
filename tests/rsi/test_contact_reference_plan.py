from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.contact_reference_plan import (
    ContactReferencePlan,
    condition_contact_reference,
)
from rosclaw_soccer.rsi.shared_nn_forecast import policy_state_hash


def plan(start=10, delta=0.2, peak=14):
    return ContactReferencePlan(start, tuple(range(29)), (delta,) + (0.0,) * 28, peak)


def test_c2_entry_peak_and_end_have_exact_positions_and_zero_derivatives():
    p, v = plan().offsets()
    np.testing.assert_array_equal(p[[0, 48]], np.zeros((2, 29)))
    np.testing.assert_array_equal(v[[0, 14, 48]], np.zeros((3, 29)))
    assert p[14, 0] == 0.2 and not p[:, 1:].any()
    assert abs(v).max() < 4
    with pytest.raises(ValueError, match="velocity cap"):
        plan(delta=0.34, peak=6).offsets()


def test_install_is_atomic_past_anchor_and_zero_branch_are_exact():
    q, v = np.zeros((100, 29), np.float32), np.zeros((100, 29), np.float32)
    policy = SimpleNamespace(
        time_step=10,
        WARMUP_STEPS=0,
        control_dt=0.02,
        runtime_mode="sim",
        use_body_frame_ball=False,
        motion_total_steps=100,
        motion_joint_pos=q,
        motion_joint_vel=v,
        anchor=object(),
    )
    limits = np.tile([-2.0, 2.0], (29, 1))
    assert not condition_contact_reference(policy, plan(delta=0.0), limits_hg=limits)["applied"]
    assert policy.motion_joint_pos is q and policy.motion_joint_vel is v
    limits[0] = [-0.1, 0.1]
    r = condition_contact_reference(policy, plan(), limits_hg=limits)
    assert not r["applied"] and policy.motion_joint_pos is q
    limits[0] = [-2.0, 2.0]
    anchor = policy.anchor
    assert condition_contact_reference(policy, plan(), limits_hg=limits)["applied"]
    np.testing.assert_array_equal(policy.motion_joint_pos[:11], q[:11])
    assert policy.anchor is anchor


def test_owned_contact_reference_future_matches_actual_nn_and_native_history(tmp_path):
    player, f = setup(tmp_path)
    player.world.advance({player.prefix: player.pending.copy()})
    player.commit()
    policy = player.policy
    policy.control_dt = 0.02
    policy.motion_total_steps = 200
    policy.motion_joint_pos = np.zeros((200, 29), np.float32)
    policy.motion_joint_vel = np.zeros((200, 29), np.float32)
    reference = plan(start=int(policy.time_step), delta=0.03)
    before = policy_state_hash(policy)
    predicted = f.forecast(controls=4, before_proposal=True, reference_plan=reference)
    assert predicted["reference_conditioned"] and policy_state_hash(policy) == before
    assert condition_contact_reference(
        policy,
        reference,
        limits_hg=player.world.model.jnt_range[player.layout.joints].astype(np.float64),
    )["applied"]
    actual, targets = [], []
    for _ in range(4):
        target, _ = player.propose()
        targets.append(target)
        actual.append(player.world.advance({player.prefix: target}))
        player.commit()
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], np.concatenate([r[key] for r in actual]))
    np.testing.assert_array_equal(predicted["joint_targets"], np.stack(targets))
