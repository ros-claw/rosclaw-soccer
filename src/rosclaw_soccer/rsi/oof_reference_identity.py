"""Check selected SIM rollouts against the independently executed paired arm."""

import numpy as np


def exact_reference_arrays(actual: dict, reference: dict) -> bool:
    """Both bodies, contact and every saved actor history must be byte-identical."""
    keys = (
        "qpos",
        "qvel",
        "torque",
        "ball_forces",
        "executed_receiver_action",
        "passer_action",
        "receiver_observation",
        "receiver_command",
        "receiver_return_action",
        "receiver_return_observation",
        "receiver_return_target",
        "passer_follow_action",
        "passer_follow_command",
        "passer_follow_control",
        "passer_follow_detector_delta",
        "passer_follow_observation",
        "passer_follow_target",
    )
    if (
        type(actual) is not dict
        or type(reference) is not dict
        or any(k not in actual or k not in reference for k in keys)
    ):
        raise ValueError("all actual shared physical and own actor histories required")
    for key in keys:
        a, b = actual[key], reference[key]
        if (
            type(a) is not np.ndarray
            or type(b) is not np.ndarray
            or a.dtype != b.dtype
            or a.shape != b.shape
        ):
            return False
        if not np.isfinite(a).all() or not np.isfinite(b).all() or a.tobytes() != b.tobytes():
            return False
    return True
