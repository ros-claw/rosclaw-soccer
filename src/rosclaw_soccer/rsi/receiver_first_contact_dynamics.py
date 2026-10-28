"""Measured receiver-contact dynamics; observations are not isolated impulses."""

import numpy as np


def first_contact_dynamics(
    qpos: np.ndarray,
    qvel: np.ndarray,
    forces: np.ndarray,
    *,
    ball_q: int,
    ball_v: int,
    receiver_q: int,
    receiver_v: int,
) -> dict:
    """Measure actual native-substep ball and root motion around first contact.

    The fixed half-second follow-up is censored when the real episode ends.
    Net velocity change includes ground and all contacts; never infer an impulse.
    """
    if (
        not isinstance(qpos, np.ndarray)
        or not isinstance(qvel, np.ndarray)
        or not isinstance(forces, np.ndarray)
        or qpos.ndim != 3
        or qpos.shape[1:] != (10, 79)
        or qvel.shape != (len(qpos), 10, 76)
        or forces.shape != (len(qpos), 10, 2, 3)
        or len(qpos) == 0
        or any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (qpos, qvel, forces))
        or np.any(forces < 0)
        or any(type(x) is not int for x in (ball_q, ball_v, receiver_q, receiver_v))
        or not 0 <= ball_q <= 76
        or not 0 <= receiver_q <= 76
        or not 0 <= ball_v <= 73
        or not 0 <= receiver_v <= 73
        or set(range(ball_q, ball_q + 3)) & set(range(receiver_q, receiver_q + 3))
        or set(range(ball_v, ball_v + 3)) & set(range(receiver_v, receiver_v + 3))
    ):
        raise ValueError("finite aligned actual shared-world substep histories required")
    q = qpos.reshape(-1, 79)
    v = qvel.reshape(-1, 76)
    f = forces[:, :, 1, :].reshape(-1, 3)
    contacts = np.flatnonzero(np.any(f > 1.0, axis=1))
    if len(contacts) == 0:
        return dict(
            first_contact_substep=None,
            first_contact_groups=[],
            precontact_observed=False,
            half_second_observed=False,
            precontact_relative_speed_mps=None,
            postcontact_relative_speed_mps=None,
            half_second_relative_speed_mps=None,
            half_second_root_ball_distance_m=None,
            half_second_ball_displacement_m=None,
        )
    first = int(contacts[0])
    groups = [
        name
        for name, value in zip(("left_foot", "right_foot", "nonfoot"), f[first], strict=True)
        if value > 1.0
    ]

    def relative_speed(i: int) -> float:
        return float(np.linalg.norm(v[i, ball_v : ball_v + 2] - v[i, receiver_v : receiver_v + 2]))

    follow = first + 250  # 250 physical steps at 2 ms; first sample is post-step.
    observed = follow < len(q)
    return dict(
        first_contact_substep=first,
        first_contact_groups=groups,
        precontact_observed=first > 0,
        half_second_observed=observed,
        precontact_relative_speed_mps=relative_speed(first - 1) if first > 0 else None,
        postcontact_relative_speed_mps=relative_speed(first),
        half_second_relative_speed_mps=relative_speed(follow) if observed else None,
        half_second_root_ball_distance_m=float(
            np.linalg.norm(q[follow, ball_q : ball_q + 2] - q[follow, receiver_q : receiver_q + 2])
        )
        if observed
        else None,
        half_second_ball_displacement_m=float(
            np.linalg.norm(q[follow, ball_q : ball_q + 2] - q[first, ball_q : ball_q + 2])
        )
        if observed
        else None,
    )
