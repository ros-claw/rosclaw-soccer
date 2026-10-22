"""Actual solver contact identity and radial direction, not an invented force.

MuJoCo solver geometry corresponds to pre-integration kinematics; canonical
q/v stored by the native runner are after the step. Keep these clocks distinct.
"""

import math
from typing import Any

import numpy as np


def summarize_contacts(events: list[dict[str, Any]], goal: np.ndarray) -> dict[str, Any]:
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("finite unchanged world goal required")
    bodies: dict[str, Any] = {}
    for event in events:
        body = event["robot_body"]
        norm = event["force_norm_N"]
        role = event["role"]
        center = np.array(event["ball_solver_center_world"], np.float64)
        point = np.array(event["contact_position_world"], np.float64)
        if (
            type(body) is not str
            or role not in (0, 1, 2)
            or not math.isfinite(norm)
            or norm <= (1 if role == 2 else 10)
            or center.shape != (3,)
            or point.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(point).all()
        ):
            raise ValueError("finite actual thresholded native solver event required")
        row = bodies.setdefault(
            body,
            dict(
                role=role,
                events=0,
                peak_norm_N=0.0,
                per_contact_norm_integral_Ns=0.0,
                first_solver_time_s=event["solver_time_s"],
                radial_force_weighted_xy=np.zeros(2),
                weighted_height=0.0,
                weight=0.0,
            ),
        )
        if row["role"] != role:
            raise ValueError("fixed compiled body contact role required")
        row["events"] += 1
        row["peak_norm_N"] = max(row["peak_norm_N"], norm)
        row["per_contact_norm_integral_Ns"] += norm * 0.002
        radial = center[:2] - point[:2]
        direction = goal[:2] - center[:2]
        if np.linalg.norm(radial) > 1e-10 and np.linalg.norm(direction) > 1e-10:
            direction /= np.linalg.norm(direction)
            radial /= np.linalg.norm(radial)
            projected = np.array(
                [radial @ direction, direction[0] * radial[1] - direction[1] * radial[0]]
            )
            row["radial_force_weighted_xy"] += norm * projected
        row["weighted_height"] += norm * (point[2] - center[2])
        row["weight"] += norm
    for row in bodies.values():
        vector = row.pop("radial_force_weighted_xy")
        row["radial_heading_degrees"] = (
            float(np.degrees(np.arctan2(vector[1], vector[0])))
            if np.linalg.norm(vector) > 1e-10
            else None
        )
        row["contact_height_relative_ball_center_m"] = row.pop("weighted_height") / row.pop(
            "weight"
        )
    return dict(
        bodies=bodies,
        nonfoot_body_names=sorted(k for k, v in bodies.items() if v["role"] == 2),
        radial_direction_is_not_measured_force_direction=True,
        solver_positions_preintegration_body_velocity_postintegration=True,
        aggregate_norm_integral_is_not_source_max_per_role_impulse=True,
    )
