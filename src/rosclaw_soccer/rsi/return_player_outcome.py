"""Causal-window proxy outcomes for SIM return-player learning.

Foot contact is only a contact proxy, never possession or task success.
"""

import numpy as np


def return_player_outcome(ball_forces, body_failed, *, sample_start):
    if (
        not isinstance(ball_forces, np.ndarray)
        or ball_forces.ndim != 4
        or ball_forces.shape[1:] != (10, 2, 3)
        or not len(ball_forces)
        or not np.isfinite(ball_forces).all()
        or (ball_forces < 0).any()
        or not isinstance(body_failed, np.ndarray)
        or body_failed.shape != (len(ball_forces), 2)
        or body_failed.dtype != np.bool_
        or type(sample_start) is not int
        or not 0 <= sample_start < len(ball_forces)
    ):
        raise ValueError("complete finite physics and sampled control boundary required")
    forces = ball_forces[sample_start:]
    failed = body_failed[sample_start:].any(axis=0)
    a_foot = bool((forces[:, :, 0, :2] > 10).any())
    a_other = bool((forces[:, :, 0, 2] > 1).any())
    b_other = bool((forces[:, :, 1, 2] > 1).any())
    both_healthy = not bool(failed.any())
    # Contact only: no claim about event order, ball control, or possession.
    clean_contact = a_foot and not a_other and both_healthy
    score = float(clean_contact - a_other - b_other - (not both_healthy))
    return {
        "sample_start": sample_start,
        "observed_controls_after_start": len(forces),
        "a_foot_contact": a_foot,
        "a_nonfoot_contact": a_other,
        "b_nonfoot_contact": b_other,
        "a_body_failed": bool(failed[0]),
        "b_body_failed": bool(failed[1]),
        "clean_a_foot_contact_proxy": clean_contact,
        "terminal_proxy_reward": score,
        "possession_proven": False,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }


def discounted_sample_returns(outcome, *, sample_end, gamma=0.99):
    """Credit terminal proxy at actual world end, including unsampled controls."""
    start = outcome["sample_start"]
    terminal = start + outcome["observed_controls_after_start"] - 1
    if (
        type(sample_end) is not int
        or not start <= sample_end <= terminal
        or not isinstance(gamma, float)
        or not 0 < gamma <= 1
        or not np.isfinite(outcome["terminal_proxy_reward"])
    ):
        raise ValueError("valid sampled interval and full physical terminal time required")
    times = np.arange(start, sample_end + 1, dtype=np.int64)
    return np.asarray(outcome["terminal_proxy_reward"] * gamma ** (terminal - times), np.float32)
