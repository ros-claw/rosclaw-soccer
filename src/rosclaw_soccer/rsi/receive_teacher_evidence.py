"""Validate actual recorded teacher composition, not task success."""

import numpy as np


def audit_teacher_torque(
    torque: np.ndarray,
    parent: np.ndarray,
    delta: np.ndarray,
    *,
    receiver_actuators: np.ndarray,
    commands: int,
    active_commands: int,
) -> dict:
    """Require exact applied/parent accounting and no teammate intervention."""
    if (
        not isinstance(torque, np.ndarray)
        or torque.ndim != 3
        or torque.shape[1] != 10
        or len(torque) < 1
        or not isinstance(parent, np.ndarray)
        or not isinstance(delta, np.ndarray)
        or parent.shape != torque.shape
        or delta.shape != torque.shape
        or not all(np.isfinite(x).all() for x in (torque, parent, delta))
        or not isinstance(receiver_actuators, np.ndarray)
        or receiver_actuators.shape != (29,)
        or receiver_actuators.dtype.kind not in "iu"
        or len(set(receiver_actuators.tolist())) != 29
        or (receiver_actuators < 0).any()
        or (receiver_actuators >= torque.shape[2]).any()
        or type(commands) is not int
        or commands != (len(torque) - 1) * 10
        or type(active_commands) is not int
        or not 0 <= active_commands <= commands
    ):
        raise ValueError("complete finite consecutive teacher torque history required")
    if not np.array_equal(torque - parent, delta) or np.any(delta[0]):
        raise ValueError("actual torque and parent composition must agree at every native substep")
    other = np.ones(torque.shape[2], dtype=bool)
    other[receiver_actuators] = False
    if np.any(delta[:, :, other] != 0):
        raise ValueError("teacher may not change another player's torque")
    changed = np.any(delta[:, :, receiver_actuators] != 0, axis=2)
    n = int(np.count_nonzero(changed))
    if n > active_commands:
        raise ValueError("changed native samples exceed reported teacher activations")
    return {
        "native_teacher_commands": commands,
        "native_active_commands": active_commands,
        "native_changed_commands": n,
        "maximum_abs_applied_delta_nm": float(np.max(np.abs(delta))),
        "other_player_torque_unchanged": True,
    }
