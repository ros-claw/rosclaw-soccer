"""Conservative native-step screening, not a possession or promotion gate."""

import numpy as np


def screen_receive(forces, qpos, qvel, body_failed, *, ball_q, ball_v, root_q, root_v):
    """Classify a measured pass-to-foot event and its observed 200ms tail.

    Thresholds identify promising data to investigate, not proof of control.
    Missing observations are explicitly censored; no future values are policy inputs.
    """
    f, q, v, failed = map(np.asarray, (forces, qpos, qvel, body_failed))
    if (
        f.ndim != 3
        or f.shape[1:] != (2, 3)
        or len(f) == 0
        or q.ndim != 2
        or v.ndim != 2
        or len(q) != len(f)
        or len(v) != len(f)
        or failed.shape != (len(f) // 10, 2)
        or len(f) % 10
        or failed.dtype != np.bool_
        or not all(np.isfinite(x).all() for x in (f, q, v))
        or np.any(f < 0)
        or any(type(x) is not int or x < 0 for x in (ball_q, ball_v, root_q, root_v))
        or ball_q + 3 > q.shape[1]
        or root_q + 3 > q.shape[1]
        or ball_v + 3 > v.shape[1]
        or root_v + 3 > v.shape[1]
    ):
        raise ValueError("complete finite 2ms two-body native history required")
    a = np.flatnonzero(np.max(f[:, 0, :2], axis=1) > 10)
    if not len(a):
        return dict(status="NO_PASSER_FOOT", screen_candidate=False)
    start = int(a[0])
    b = np.flatnonzero(np.max(f[start:, 1, :2], axis=1) > 10) + start
    if not len(b):
        return dict(status="NO_RECEIVER_FOOT", passer_step=start, screen_candidate=False)
    touch = int(b[0])
    end = touch + 100
    if end >= len(f):
        return dict(
            status="POST_CONTACT_CENSORED",
            passer_step=start,
            receiver_step=touch,
            screen_candidate=False,
        )
    passer_nonfoot = bool(np.any(f[start : end + 1, 0, 2] > 1))
    receiver_nonfoot = bool(np.any(f[start : end + 1, 1, 2] > 1))
    healthy = not bool(failed[: end // 10 + 1].any())
    distance = float(np.linalg.norm(q[end, ball_q : ball_q + 3] - q[end, root_q : root_q + 3]))
    speed = float(np.linalg.norm(v[end, ball_v : ball_v + 3] - v[end, root_v : root_v + 3]))
    candidate = (
        not passer_nonfoot
        and not receiver_nonfoot
        and healthy
        and distance <= 0.65
        and speed <= 0.5
    )
    return dict(
        status="SCREEN_CANDIDATE" if candidate else "SCREEN_REJECTED",
        passer_step=start,
        receiver_step=touch,
        passer_nonfoot=passer_nonfoot,
        receiver_nonfoot=receiver_nonfoot,
        body_healthy=healthy,
        ball_root_distance_0p2s_m=distance,
        relative_speed_0p2s_mps=speed,
        screen_candidate=bool(candidate),
    )
