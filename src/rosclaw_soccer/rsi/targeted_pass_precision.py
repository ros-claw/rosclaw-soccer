"""New TRAIN pass-delivery objective; never changes old first-touch grading.

One-body reachability pilot only. A point crossing does not prove B receives
the ball. Formal arrival window is300 controls, with separately observed
100-control neural recovery required for teacher selection success.
"""

from typing import Any

import numpy as np


def targeted_pass_precision(
    qpos: np.ndarray,
    qvel: np.ndarray,
    forces: np.ndarray,
    goal: np.ndarray,
    *,
    body_healthy: bool,
    recovery_observed: bool,
) -> dict[str, Any]:
    n = len(qpos)
    if (
        qpos.shape != (n, 43)
        or qvel.shape != (n, 41)
        or forces.shape != (n, 3)
        or n < 1
        or goal.shape != (3,)
        or any(not np.isfinite(a).all() for a in (qpos, qvel, forces, goal))
        or (forces < 0).any()
        or type(body_healthy) is not bool
        or type(recovery_observed) is not bool
    ):
        raise ValueError("finite canonical native trajectory and actual whole recovery required")
    formal = min(n, 3000)
    contacts = np.flatnonzero(forces[:formal, :2].max(axis=1) > 10)
    first = int(contacts[0]) if len(contacts) else None
    error = speed = arrival = None
    velocity = None
    if first is not None:
        points = qpos[first:formal, 36:39]
        local = int(np.argmin(np.linalg.norm(points - goal, axis=1)))
        arrival = first + local
        error = float(np.linalg.norm(qpos[arrival, 36:39] - goal))
        velocity = qvel[arrival, 35:38].copy()
        speed = float(np.linalg.norm(velocity))
    clean = not bool((forces[:, 2] > 1).any())
    success = bool(
        body_healthy
        and recovery_observed
        and clean
        and error is not None
        and error <= 0.1
        and speed is not None
        and speed <= 3.0
    )
    return dict(
        first_foot_contact_substep=first,
        formal_native_steps=formal,
        closest_goal_substep=arrival,
        minimum_goal_error_3d_m=error,
        arrival_speed_mps=speed,
        arrival_velocity_world=None if velocity is None else velocity.tolist(),
        whole_body_healthy=body_healthy,
        recovery_observed=recovery_observed,
        clean_foot=clean,
        precision_0p1m_moderate_speed_success=success,
        received_by_teammate_verified=False,
        old_first_touch_exam_regraded=False,
    )


def precision_rank(result: dict[str, Any]) -> tuple[int, int, int, float]:
    error, speed = result["minimum_goal_error_3d_m"], result["arrival_speed_mps"]
    score = -1e6 if error is None else -error - 0.03 * max(0.0, speed - 3.0) ** 2
    return (
        int(result["whole_body_healthy"] and result["recovery_observed"]),
        int(result["clean_foot"]),
        int(result["first_foot_contact_substep"] is not None),
        score,
    )
