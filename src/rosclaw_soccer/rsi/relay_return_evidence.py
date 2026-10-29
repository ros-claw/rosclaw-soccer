"""Post-return observations; a target hit is not a teammate possession claim."""

import numpy as np


def return_observation(qpos, qvel, forces, *, start, goal, ball_q, ball_v, teammate_q):
    """Observe the closest target approach and subsequent actual teammate contact.

    This is descriptive evidence, not a success or refereeing gate. Offsets use
    native model ordering, and ``start`` is a native-step index.
    """
    q, v, f, goal = map(np.asarray, (qpos, qvel, forces, goal))
    if (
        q.ndim != 2
        or v.ndim != 2
        or f.shape != (len(q), 2, 3)
        or len(v) != len(q)
        or goal.shape != (3,)
        or not all(np.isfinite(x).all() for x in (q, v, f, goal))
        or np.any(f < 0)
        or type(start) is not int
        or not 0 <= start < len(q)
        or any(type(x) is not int or x < 0 for x in (ball_q, ball_v, teammate_q))
        or max(ball_q, teammate_q) + 3 > q.shape[1]
        or ball_v + 3 > v.shape[1]
    ):
        raise ValueError("finite, aligned native trajectory and valid addresses required")
    ball = q[:, ball_q : ball_q + 3]
    index = start + int(np.argmin(np.linalg.norm(ball[start:] - goal, axis=1)))
    later = f[index:, 0, :]

    def first(mask):
        found = np.flatnonzero(mask)
        return None if not len(found) else index + int(found[0])

    return dict(
        closest_step=index,
        closest_error_m=float(np.linalg.norm(ball[index] - goal)),
        teammate_planar_distance_m=float(
            np.linalg.norm(ball[index, :2] - q[index, teammate_q : teammate_q + 2])
        ),
        ball_speed_mps=float(np.linalg.norm(v[index, ball_v : ball_v + 3])),
        teammate_first_foot_over_10n_step=first(np.max(later[:, :2], axis=1) > 10),
        teammate_first_nonfoot_over_1n_step=first(later[:, 2] > 1),
        teammate_nonfoot_peak_n=float(np.max(later[:, 2])),
        teammate_possession_proven=False,
    )
