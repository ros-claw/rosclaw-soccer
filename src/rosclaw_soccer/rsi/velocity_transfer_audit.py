"""Shape/summary checks for an independent native velocity transfer replay."""

from typing import Any

import numpy as np


def validate_velocity_raw(raw: dict[str, np.ndarray]) -> int:
    n = len(raw["qpos_before"])
    widths = dict(
        qpos_before=(43,),
        qvel_before=(41,),
        observation=(98,),
        action=(29,),
        target=(29,),
        command_body=(3,),
        command_world=(3,),
        omega=(3,),
        phase=(),
        qpos=(10, 43),
        qvel=(10, 41),
        torque=(10, 29),
        force=(10, 29),
        ball_force=(10,),
    )
    floats = {"observation", "action", "command_body", "phase"}
    if (
        not 1 <= n <= 300
        or set(raw) != set(widths)
        or any(
            raw[k].shape != (n, *shape)
            or raw[k].dtype != (np.float32 if k in floats else np.float64)
            or not np.isfinite(raw[k]).all()
            for k, shape in widths.items()
        )
    ):
        raise ValueError("all bounded finite exact native/ownNN transfer arrays required")
    if not np.array_equal(raw["qpos_before"][1:], raw["qpos"][:-1, -1]) or not np.array_equal(
        raw["qvel_before"][1:], raw["qvel"][:-1, -1]
    ):
        raise ValueError("no physical reset or boundary state splice allowed")
    return n


def summarize_velocity_raw(
    raw: dict[str, np.ndarray], case: int, initial: str, speed: float
) -> dict[str, Any]:
    n = validate_velocity_raw(raw)
    upright = 1 - 2 * (raw["qpos"][:, :, 4] ** 2 + raw["qpos"][:, :, 5] ** 2)
    height = float(raw["qpos"][:, :, 2].min())
    return dict(
        case=case,
        initial=initial,
        speed_mps=speed,
        controls=n,
        body_stable=bool(n == 300 and height >= 0.65 and upright.min() >= 0.9),
        minimum_height_m=height,
        minimum_upright=float(upright.min()),
        pulse_mean_velocity_mps=raw["qvel_before"][100:150, :2].mean(0).tolist()
        if n >= 150
        else None,
        stop_drift_m=float(np.linalg.norm(raw["qpos_before"][-1, :2] - raw["qpos_before"][150, :2]))
        if n == 300
        else None,
        maximum_robot_ball_force_N=float(raw["ball_force"].max()),
        native_steps=n * 10,
        neural_calls=n,
    )
