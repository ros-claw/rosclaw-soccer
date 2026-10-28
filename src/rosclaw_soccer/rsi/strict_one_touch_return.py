"""SIM evidence for a direct physical return distinct from trapping."""

import numpy as np


def strict_one_touch_return(
    ball_xyz: np.ndarray,
    ball_velocity_xy: np.ndarray,
    forces: np.ndarray,
    body_failed: np.ndarray,
    *,
    target_xyz: np.ndarray,
    kick_origin_xy: np.ndarray,
    earliest_return_substep: int,
) -> dict:
    """Require actual A then B feet, healthy bodies, and exact return target.

    Input rows are native substeps, not replayed balls or reconstructed paths.
    This reports a direct return independently of the 15-control trap test.
    """
    n = len(ball_xyz)
    if (
        n < 1
        or ball_xyz.shape != (n, 3)
        or ball_velocity_xy.shape != (n, 2)
        or forces.shape != (n, 2, 3)
        or body_failed.ndim != 2
        or body_failed.shape[1] != 2
        or len(body_failed) * 10 != n
        or body_failed.dtype != np.bool_
        or target_xyz.shape != (3,)
        or kick_origin_xy.shape != (2,)
        or type(earliest_return_substep) is not int
        or not 0 <= earliest_return_substep < n
        or any(
            not np.isfinite(x).all()
            for x in (ball_xyz, ball_velocity_xy, forces, target_xyz, kick_origin_xy)
        )
        or (forces < 0).any()
    ):
        raise ValueError(
            "complete actual finite ball, force, two-body and target histories required"
        )
    direction = target_xyz[:2] - kick_origin_xy
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError("nondegenerate real return direction required")
    direction /= length
    a = np.flatnonzero(forces[:, 0, :2].max(axis=1) > 10)
    b = np.flatnonzero((forces[:, 1, :2].max(axis=1) > 10) & (ball_velocity_xy @ direction >= 1))
    first_a = int(a[0]) if len(a) else None
    eligible = (
        b[(b > first_a) & (b >= earliest_return_substep)]
        if first_a is not None
        else np.array([], dtype=int)
    )
    first_b = int(eligible[0]) if len(eligible) else None
    no_receiver_before_pass = first_a is not None and not (forces[: first_a + 1, 1, :] > 1).any()
    min_error = (
        float(np.linalg.norm(ball_xyz[first_b:] - target_xyz, axis=1).min())
        if first_b is not None
        else None
    )
    clean = not (forces[:, :, 2] > 1).any()
    healthy = not body_failed.any()
    return {
        "actual_passer_foot_substep": first_a,
        "actual_direct_return_substep": first_b,
        "return_target_min_error_m": min_error,
        "both_nonfoot_clear": bool(clean),
        "both_bodies_healthy": bool(healthy),
        "receiver_no_contact_before_pass": bool(no_receiver_before_pass),
        "strict_one_touch_return": bool(
            first_b is not None
            and min_error is not None
            and min_error <= 0.1
            and clean
            and healthy
            and no_receiver_before_pass
        ),
        "one_touch_is_not_original_trap": True,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
