"""SIM-only velocity-prior target decoding on an unchanged torque body.

Different controller PD gains are explicit inputs, not edits to mjModel.
This adapter is not a hardware driver and cannot open any transport.
"""

from typing import Any

import numpy as np


def require_torque_encoding(model: Any, layout: Any) -> np.ndarray:
    ai, ji = layout.ai, layout.joints
    if (
        not np.array_equal(model.actuator_gainprm[ai, 0], np.ones(29))
        or model.actuator_gainprm[ai, 1:].any()
        or model.actuator_biasprm[ai].any()
        or model.actuator_gaintype[ai].any()
        or model.actuator_biastype[ai].any()
        or not np.array_equal(model.actuator_gear[ai, 0], np.ones(29))
        or model.actuator_gear[ai, 1:].any()
        or not model.jnt_actfrclimited[ji].all()
    ):
        raise ValueError(
            "unchanged unit-gain torque actuators and actual joint force caps required"
        )
    ranges = np.array(model.jnt_actfrcrange[ji], np.float64)
    if (
        ranges.shape != (29, 2)
        or not np.isfinite(ranges).all()
        or not (ranges[:, 1] > 0).all()
        or not np.array_equal(ranges[:, 0], -ranges[:, 1])
    ):
        raise ValueError("finite symmetric compiled effort limits required")
    return ranges[:, 1].copy()


def decode_pd_torque(
    target: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    effort: np.ndarray,
) -> np.ndarray:
    arrays = (target, q, v, kp, kd, effort)
    if any(
        not isinstance(a, np.ndarray)
        or a.shape != (29,)
        or a.dtype != np.float64
        or not np.isfinite(a).all()
        for a in arrays
    ) or any(not (a > 0).all() for a in (kp, kd, effort)):
        raise ValueError("finite double29 software PD targets/state/gains/caps required")
    return np.asarray(np.clip(kp * (target - q) - kd * v, -effort, effort), dtype=np.float64)


def transfer_cases() -> tuple[tuple[str, float], ...]:
    return tuple(
        (initial, speed)
        for initial in ("velocity_home", "recorded_paid")
        for speed in (0.0, 0.3, 0.6, 1.0)
    )


def world_pulse(speed: float, yaw: float) -> tuple[float, float, float]:
    if type(speed) is not float or type(yaw) is not float or not np.isfinite([speed, yaw]).all():
        raise ValueError("explicit finite immutable command scalars required")
    return (float(speed * np.cos(yaw)), float(speed * np.sin(yaw)), 0.0)
