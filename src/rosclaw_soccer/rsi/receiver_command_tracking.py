"""Observed command-to-body response for SIM receiving histories.

The velocity target is an intent, not a promised actuator trajectory. These
descriptive errors do not identify the cause or prove ball reachability.
"""

import numpy as np


def receiver_command_tracking(
    command: np.ndarray,
    root_quaternion: np.ndarray,
    root_velocity_world: np.ndarray,
    past_contact: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare commands with observed post-control velocity in body yaw frame."""
    if not all(
        isinstance(a, np.ndarray)
        for a in (command, root_quaternion, root_velocity_world, past_contact)
    ):
        raise ValueError("aligned actual arrays required")
    n = len(command)
    if (
        n < 1
        or command.shape != (n, 3)
        or root_quaternion.shape != (n, 4)
        or root_velocity_world.shape != (n, 3)
        or past_contact.shape != (n, 2)
        or past_contact.dtype != np.bool_
        or any(not np.isfinite(a).all() for a in (command, root_quaternion, root_velocity_world))
        or not np.allclose(np.linalg.norm(root_quaternion, axis=1), 1, rtol=0, atol=1e-5)
        or np.any(np.diff(past_contact.astype(np.int8), axis=0) < 0)
        or (
            eligible is not None
            and (
                not isinstance(eligible, np.ndarray)
                or eligible.shape != (n,)
                or eligible.dtype != np.bool_
            )
        )
    ):
        raise ValueError(
            "finite sequential command, unit orientation and persistent contact required"
        )
    w, x, y, z = root_quaternion.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = root_velocity_world[:, 0], root_velocity_world[:, 1]
    body_velocity = np.stack((c * vx + s * vy, -s * vx + c * vy), axis=1)
    error = np.linalg.norm(command[:, :2] - body_velocity, axis=1)
    pre = ~past_contact.any(axis=1)
    if eligible is not None:
        pre &= eligible
    command_speed = np.linalg.norm(command[:, :2], axis=1)
    body_speed = np.linalg.norm(body_velocity, axis=1)
    return {
        "observations": n,
        "precontact_observations": int(pre.sum()),
        "precontact_command_speed_mean_mps": float(command_speed[pre].mean())
        if pre.any()
        else None,
        "precontact_body_speed_mean_mps": float(body_speed[pre].mean()) if pre.any() else None,
        "precontact_velocity_error_mean_mps": float(error[pre].mean()) if pre.any() else None,
        "precontact_velocity_error_p90_mps": float(np.quantile(error[pre], 0.9))
        if pre.any()
        else None,
        "same_tick_post_control_observation": True,
        "error_does_not_prove_interface_failure": True,
        "reachability_proven": False,
    }
