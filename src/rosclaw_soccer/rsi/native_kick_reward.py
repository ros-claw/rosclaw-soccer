"""Causal, native-step football feedback; no deployment observation is augmented.

Reward at t depends only on the real before/after states of that transition.
Full returns use future rewards as training labels, never as policy inputs.
"""

import numpy as np


def native_kick_rewards(
    *,
    ball_positions,
    pelvis_after,
    joint_velocity_after,
    applied_torque,
    hard_limits,
    foot_contact,
    illegal_contact,
    used_parent,
    dt=0.002,
):
    ball, pelvis, velocity, torque, limits, foot, illegal, fallback = map(
        np.asarray,
        (
            ball_positions,
            pelvis_after,
            joint_velocity_after,
            applied_torque,
            hard_limits,
            foot_contact,
            illegal_contact,
            used_parent,
        ),
    )
    n = len(torque)
    if (
        torque.ndim != 2
        or not n
        or ball.shape != (n + 1, 3)
        or pelvis.shape != (n, 7)
        or velocity.shape != torque.shape
        or limits.shape != (torque.shape[1],)
        or any(v.shape != (n,) for v in (foot, illegal, fallback))
        or any(
            not np.isfinite(v).all()
            for v in (ball, pelvis, velocity, torque, limits, foot, illegal, fallback)
        )
        or np.any(limits <= 0)
        or not np.isfinite(dt)
        or dt != 0.002
    ):
        raise ValueError("finite aligned native 2ms state/action/contact feedback required")
    quaternions = pelvis[:, 3:7]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("actual normalized pelvis orientations required")
    seen_contact = np.maximum.accumulate(foot.astype(bool))
    distance = np.linalg.norm(ball - [5.0, 0.0, 0.11], axis=1)
    progress = seen_contact * np.clip(distance[:-1] - distance[1:], -0.05, 0.05)
    upright = 1 - 2 * (quaternions[:, 1] ** 2 + quaternions[:, 2] ** 2)
    fell = (pelvis[:, 2] < 0.65) | (upright < 0.9)
    effort = np.mean((torque / limits) ** 2, axis=1)
    motion = np.mean((velocity / 10.0) ** 2, axis=1)
    rewards = progress - dt * (0.5 * (1 - upright) + 0.01 * motion + 0.05 * effort)
    crossings = np.flatnonzero((ball[:-1, 0] < 5) & (ball[1:, 0] >= 5) & seen_contact)
    crossing_index, error = None, None
    if len(crossings):
        crossing_index = int(crossings[0])
        k = crossing_index
        fraction = (5 - ball[k, 0]) / (ball[k + 1, 0] - ball[k, 0])
        point = ball[k] + fraction * (ball[k + 1] - ball[k])
        error = float(np.linalg.norm(point[1:] - [0.0, 0.11]))
        rewards[k] += 2.0 / (1.0 + (error / 0.25) ** 2)
    # Failure signals count at their first real occurrence, not retroactively.
    for event, penalty in ((fell, 5.0), (illegal.astype(bool), 2.0)):
        indices = np.flatnonzero(event)
        if len(indices):
            rewards[indices[0]] -= penalty
    constraints = np.maximum(illegal.astype(float), fallback.astype(float))
    return dict(
        rewards=rewards.astype(np.float32),
        fall_costs=fell.astype(np.float32),
        constraint_costs=constraints.astype(np.float32),
        crossing_index=crossing_index,
        crossing_error_m=error,
    )


def finite_returns(rewards, *, gamma):
    value = np.asarray(rewards, dtype=np.float64)
    if (
        value.ndim != 1
        or not len(value)
        or not np.isfinite(value).all()
        or not np.isfinite(gamma)
        or not 0 < gamma <= 1
    ):
        raise ValueError("finite bounded discount and nonempty aligned rewards required")
    result = np.empty_like(value)
    running = 0.0
    for i in range(len(value) - 1, -1, -1):
        running = value[i] + gamma * running
        result[i] = running
    return result.astype(np.float32)
