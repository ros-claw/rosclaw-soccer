"""Measure actual return-transition targets and outcomes, not causality."""

import numpy as np


def return_transition_diagnostics(
    targets: np.ndarray,
    qpos: np.ndarray,
    body_failed: np.ndarray,
    goal: np.ndarray,
    *,
    handoff_control: int,
    ball_q: int,
) -> dict:
    if (
        type(targets) is not np.ndarray
        or targets.dtype != np.float64
        or targets.ndim != 2
        or targets.shape[1] != 29
        or not np.isfinite(targets).all()
        or type(qpos) is not np.ndarray
        or qpos.dtype != np.float64
        or qpos.shape != (len(targets), 10, 79)
        or not np.isfinite(qpos).all()
        or type(body_failed) is not np.ndarray
        or body_failed.dtype != np.bool_
        or body_failed.shape != (len(targets), 2)
        or type(goal) is not np.ndarray
        or goal.dtype != np.float64
        or goal.shape != (3,)
        or not np.isfinite(goal).all()
        or type(handoff_control) is not int
        or not 1 <= handoff_control < len(targets)
        or type(ball_q) is not int
        or not 0 <= ball_q <= 76
    ):
        raise ValueError("finite aligned original SIM physical history and handoff required")
    delta = np.max(np.abs(np.diff(targets, axis=0)), axis=1)
    initial_jump = float(delta[handoff_control - 1])
    available = min(20, len(delta) - (handoff_control - 1))
    failed_at = np.flatnonzero(np.any(body_failed, axis=1))
    ball = qpos[handoff_control:, :, ball_q : ball_q + 3].reshape(-1, 3)
    return dict(
        handoff_control=handoff_control,
        handoff_target_jump_rad=initial_jump,
        following_window_controls=int(available),
        following_window_max_target_jump_rad=float(
            np.max(delta[handoff_control - 1 : handoff_control - 1 + available])
        ),
        first_body_failure_control=None if len(failed_at) == 0 else int(failed_at[0]),
        minimum_return_goal_error_m=float(np.linalg.norm(ball - goal, axis=1).min()),
        proof_of_causation=False,
    )
