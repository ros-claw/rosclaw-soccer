"""Descriptive native return-contact windows, not possession or success."""

import numpy as np


def longest_true(mask):
    run = best = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def return_interaction(
    q, v, forces, failure, *, start, direction, ball_q, ball_v, player_q, player_v
):
    q, v, f, failed, direction = map(np.asarray, (q, v, forces, failure, direction))
    if (
        q.ndim != 2
        or v.ndim != 2
        or len(q) != len(v)
        or f.shape != (len(q), 2, 3)
        or failed.dtype != np.bool_
        or failed.shape != (len(q) // 10, 2)
        or len(q) % 10
        or type(start) is not int
        or not 0 <= start < len(q)
        or direction.shape != (2,)
        or any(type(x) is not int or x < 0 for x in (ball_q, ball_v, player_q, player_v))
        or max(ball_q, player_q) + 3 > q.shape[1]
        or max(ball_v, player_v) + 3 > v.shape[1]
        or not all(np.isfinite(x).all() for x in (q, v, f, direction))
        or np.any(f < 0)
        or np.linalg.norm(direction) <= 0
    ):
        raise ValueError("finite complete native two-body return history required")
    unit = direction / np.linalg.norm(direction)
    strike = np.flatnonzero(
        (np.max(f[start:, 1, :2], axis=1) > 10) & ((v[start:, ball_v : ball_v + 2] @ unit) >= 1)
    )
    if not strike.size:
        return dict(
            directed_strike_step=None,
            first_A_foot_step=None,
            first_A_nonfoot_step=None,
            near_healthy_longest_substeps=0,
            observed_after_contact_substeps=0,
        )
    t = start + int(strike[0])
    foot = np.flatnonzero(np.max(f[t:, 0, :2], axis=1) > 10)
    nonfoot = np.flatnonzero(f[t:, 0, 2] > 1)
    contact = np.flatnonzero((f[t:, 0, :] > 1).any(axis=1))
    result = dict(
        directed_strike_step=t,
        first_A_foot_step=t + int(foot[0]) if foot.size else None,
        first_A_nonfoot_step=t + int(nonfoot[0]) if nonfoot.size else None,
        near_healthy_longest_substeps=0,
        observed_after_contact_substeps=0,
    )
    if contact.size:
        first = t + int(contact[0])
        dist = np.linalg.norm(
            q[first:, ball_q : ball_q + 2] - q[first:, player_q : player_q + 2], axis=1
        )
        speed = np.linalg.norm(
            v[first:, ball_v : ball_v + 3] - v[first:, player_v : player_v + 3], axis=1
        )
        healthy = ~np.repeat(failed.any(axis=1), 10)[first:]
        result["near_healthy_longest_substeps"] = longest_true(
            (dist <= 0.65) & (speed <= 0.6) & healthy
        )
        result["observed_after_contact_substeps"] = len(dist)
    return result
