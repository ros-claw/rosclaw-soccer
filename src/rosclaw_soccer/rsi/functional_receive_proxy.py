"""SIM research shaping for contact-followed-by-near-ball behavior.

Only an offline auxiliary reward: aggregate force has no body-part identity,
2-D finite-difference velocity is not a full ball state, and the signal does
not establish possession, a legal touch, a successful pass, or safety.
"""

import numpy as np


def functional_receive_proxy(receiver_xy, ball_xy, ball_forces, body_failed, *, eventual_failure):
    """Give one causal window bonus and preserve an observed terminal failure cost."""
    r, b, f, failed = map(np.asarray, (receiver_xy, ball_xy, ball_forces, body_failed))
    n = len(r)
    if (
        n < 1
        or r.shape != (n, 2)
        or b.shape != (n, 2)
        or f.shape != (n, 10, 3)
        or failed.shape != (n,)
        or failed.dtype != np.bool_
        or type(eventual_failure) is not bool
        or not all(np.isfinite(x).all() for x in (r, b, f))
        or np.any(f < 0)
    ):
        raise ValueError("complete aligned finite native feedback required")
    reward = np.zeros(n, dtype=np.float32)
    offset = b - r
    speed = np.full(n, np.inf)
    speed[1:] = np.linalg.norm(np.diff(offset, axis=0), axis=1) / 0.02
    seen = False
    consecutive = 0
    window = None
    for i in range(n):
        seen = seen or bool(np.max(f[i]) > 1)
        good = seen and not failed[i] and np.linalg.norm(offset[i]) <= 0.65 and speed[i] <= 0.6
        consecutive = consecutive + 1 if good else 0
        if consecutive >= 10 and window is None:
            window = i
            reward[i] += np.float32(1.0)
    if eventual_failure:
        reward[-1] -= np.float32(2.0)
    return reward, window
