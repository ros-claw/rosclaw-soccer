"""Read-only, sampled physical diagnostics of a failed relay handoff.

This does not alter the original online predicate or assign causal blame.
"""

import numpy as np


def diagnose_no_handoff(
    qpos: np.ndarray,
    qvel: np.ndarray,
    ball_forces: np.ndarray,
    body_failed: np.ndarray,
    *,
    receiver_q: int,
    receiver_v: int,
    ball_q: int,
    ball_v: int,
    excluded_contact: bool,
) -> dict:
    n = len(qpos)
    if (
        qpos.shape != (300, 10, 79)
        or qvel.shape != (300, 10, 76)
        or ball_forces.shape != (300, 10, 2, 3)
        or body_failed.shape != (300, 2)
        or body_failed.dtype != np.bool_
        or any(not np.isfinite(a).all() for a in (qpos, qvel, ball_forces))
        or np.any(ball_forces < 0)
        or type(excluded_contact) is not bool
        or any(type(x) is not int for x in (receiver_q, receiver_v, ball_q, ball_v))
        or not 0 <= receiver_q <= 72
        or not 0 <= ball_q <= 72
        or not 0 <= receiver_v <= 70
        or not 0 <= ball_v <= 70
        or set(range(receiver_q, receiver_q + 7)) & set(range(ball_q, ball_q + 7))
        or set(range(receiver_v, receiver_v + 6)) & set(range(ball_v, ball_v + 6))
    ):
        raise ValueError("exact finite H300 shared physical trajectory required")
    q, v = qpos[:, -1], qvel[:, -1]
    root, ball = q[:, receiver_q : receiver_q + 7], q[:, ball_q : ball_q + 7]
    if not np.allclose(np.linalg.norm(root[:, 3:7], axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("unit measured receiver orientation required")
    delta = ball[:, :3] - root[:, :3]
    distance = np.linalg.norm(delta[:, :2], axis=1)
    relative_speed = np.linalg.norm(
        v[:, ball_v : ball_v + 2] - v[:, receiver_v : receiver_v + 2], axis=1
    )
    w, x, y, z = root[:, 3:7].T
    front = (
        (1 - 2 * (y * y + z * z)) * delta[:, 0]
        + 2 * (x * y + w * z) * delta[:, 1]
        + 2 * (x * z - w * y) * delta[:, 2]
    )
    force = ball_forces.max(axis=1)
    kick = force[:, 0, :2].max(axis=1) > 10
    touch = force[:, 1, :].max(axis=1) > 1
    # The archived excluded-body set is episode-wide, not per-control.  A
    # nonempty set therefore prevents exact reconstruction, rather than
    # pretending we know when excluded contact happened.
    seen = np.maximum.accumulate(touch)
    tests = dict(
        contact=seen,
        distance=distance <= 0.65,
        relative_speed=relative_speed <= 0.6,
        ball_height=ball[:, 2] <= 0.3,
        ball_in_front=front >= 0.05,
        body_height=root[:, 2] >= 0.65,
        body_upright=1 - 2 * (root[:, 4] ** 2 + root[:, 5] ** 2) >= 0.9,
        body_healthy=~body_failed.any(axis=1),
    )
    good = np.logical_and.reduce(list(tests.values()))
    best = current = 0
    for value in good:
        current = current + 1 if value else 0
        best = max(best, current)
    first_kick = np.flatnonzero(kick)
    first_touch = np.flatnonzero(touch)
    touch_index = int(first_touch[0]) if len(first_touch) else None
    receiver_substeps = ball_forces[:, :, 1, :].reshape(-1, 3)
    first_substeps = np.flatnonzero(np.any(receiver_substeps > 1, axis=1))
    first_substep = int(first_substeps[0]) if len(first_substeps) else None
    first_contact_groups = (
        [
            name
            for name, value in zip(
                ("left_foot", "right_foot", "nonfoot"),
                receiver_substeps[first_substep],
                strict=True,
            )
            if value > 1
        ]
        if first_substep is not None
        else []
    )
    pre = slice(int(first_kick[0]), n) if len(first_kick) else slice(0, n)
    return dict(
        actual_passer_foot_over_10n=bool(len(first_kick)),
        first_passer_foot_control=int(first_kick[0]) if len(first_kick) else None,
        actual_receiver_contact_over_1n=bool(len(first_touch)),
        first_receiver_contact_control=int(first_touch[0]) if len(first_touch) else None,
        first_receiver_contact_substep=first_substep,
        first_receiver_contact_force_groups=first_contact_groups,
        contact_groups_are_not_individual_nonfoot_body_names=True,
        relative_speed_at_first_contact_mps=(
            float(relative_speed[touch_index]) if touch_index is not None else None
        ),
        root_ball_distance_at_first_contact_m=(
            float(distance[touch_index]) if touch_index is not None else None
        ),
        minimum_post_contact_relative_speed_mps=(
            float(relative_speed[touch_index:].min()) if touch_index is not None else None
        ),
        receiver_contact_precedes_passer_kick=bool(
            len(first_touch) and (not len(first_kick) or first_touch[0] < first_kick[0])
        ),
        excluded_contact_ever=excluded_contact,
        minimum_root_ball_xy_m=float(distance[pre].min()),
        closest_root_ball_control=int(
            np.argmin(distance[pre]) + (int(first_kick[0]) if len(first_kick) else 0)
        ),
        nearest_relative_speed_mps=float(
            relative_speed[
                np.argmin(distance[pre]) + (int(first_kick[0]) if len(first_kick) else 0)
            ]
        ),
        best_consecutive_candidate_controls=int(best),
        predicate_reconstruction_exact=not excluded_contact,
        controls_failing_each_predicate={k: int((~val).sum()) for k, val in tests.items()},
        post_touch_controls_failing_each_predicate={
            k: int((~val & seen).sum()) for k, val in tests.items()
        },
        no_handoff_reconstruction_consistent=bool(excluded_contact or best < 15),
    )
