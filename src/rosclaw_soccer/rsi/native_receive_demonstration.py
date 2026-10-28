"""Strict observed native receiving windows, not an autonomous skill certificate."""

import numpy as np


def qualify_receive_window(
    *, qpos, qvel, forces, valid, terminated, ball_q, ball_v, root_q, root_v
):
    """Require complete observed motion, foot-first and 100 clean native samples.

    This filters already-collected training demonstrations. It does not use
    the future at inference, claim a successful match, or prove generalization.
    """
    if (
        qpos.ndim != 3
        or qvel.ndim != 3
        or qpos.shape[:2] != qvel.shape[:2]
        or qpos.shape[1] != 10
        or forces.shape != (*qpos.shape[:2], 3)
        or valid.shape != (len(qpos),)
        or terminated.shape != valid.shape
        or valid.dtype != np.bool_
        or terminated.dtype != np.bool_
        or not all(np.isfinite(x).all() for x in (qpos, qvel, forces))
        or np.any(forces < 0)
        or not all(type(i) is int and i >= 0 for i in (ball_q, ball_v, root_q, root_v))
        or max(ball_q, root_q) + 3 > qpos.shape[-1]
        or max(ball_v, root_v) + 3 > qvel.shape[-1]
    ):
        raise ValueError("finite aligned 10-substep native observations required")
    if not valid.all() or terminated.any():
        return {"qualified": False, "reason": "incomplete_or_body_guard"}
    q = qpos.reshape(-1, qpos.shape[-1])
    v = qvel.reshape(-1, qvel.shape[-1])
    f = forces.reshape(-1, 3)
    foot = np.flatnonzero(np.max(f[:, :2], axis=1) > 10.0)
    nonfoot = np.flatnonzero(f[:, 2] > 1.0)
    if not len(foot):
        return {"qualified": False, "reason": "no_strong_foot"}
    if len(nonfoot):
        return {"qualified": False, "reason": "nonfoot_contact"}
    separation = np.linalg.norm(q[:, ball_q : ball_q + 2] - q[:, root_q : root_q + 2], axis=1)
    speed = np.linalg.norm(v[:, ball_v : ball_v + 3] - v[:, root_v : root_v + 3], axis=1)
    controlled = (separation <= 0.65) & (speed <= 0.5)
    run = longest = 0
    end = None
    for index in range(int(foot[0]) + 1, len(controlled)):
        run = run + 1 if controlled[index] else 0
        if run > longest:
            longest, end = run, index
    return {
        "qualified": longest >= 100,
        "reason": "observed_clean_control" if longest >= 100 else "no_continuous_control",
        "first_foot_native_step": int(foot[0]),
        "longest_controlled_native_samples": longest,
        "longest_controlled_end_native_step": end,
    }
