"""Passive evidence at the actual shared native control boundary, SIM only."""

from typing import Any

import numpy as np


def body_contact_witness(world: Any) -> dict[str, Any]:
    """Copy exact body names and per-control peak forces; never change physics."""
    world._require_current_boundary(allow_failed=True)
    return {
        prefix: {name: float(force) for name, force in bodies.items()}
        for prefix, bodies in world.last_ball_contact_bodies.items()
    }


def readiness_witness(boundary: Any, ready: bool) -> dict[str, Any]:
    """Record gate inputs after its normal update, without driving the gate."""
    world = boundary.world
    world._require_current_boundary()
    if boundary.last_control != world.control_steps or type(ready) is not bool:
        raise ValueError("updated receiving boundary and actual boolean required")
    q, v = world.layouts[boundary.prefix].canonical(world.data)
    delta = q[36:39] - q[:3]
    qw, x, y, z = q[3:7]
    front = float(
        np.array([1 - 2 * (y * y + z * z), 2 * (x * y + qw * z), 2 * (x * z - qw * y)]) @ delta
    )
    distance = float(np.linalg.norm(delta[:2]))
    relative_speed = float(np.linalg.norm(v[35:37] - v[:2]))
    upright = float(1 - 2 * (q[4] ** 2 + q[5] ** 2))
    good = bool(
        boundary.contact_seen
        and not boundary.excluded_contact
        and not world.failed
        and distance <= 0.65
        and relative_speed <= 0.6
        and q[38] <= 0.3
        and front >= 0.05
        and q[2] >= 0.65
        and upright >= 0.9
    )
    if ready != (good and boundary.consecutive >= 15):
        raise ValueError("recorded predicates disagree with actual online boundary")
    return dict(
        control=world.control_steps,
        contact_seen=boundary.contact_seen,
        excluded_contact=boundary.excluded_contact,
        consecutive=boundary.consecutive,
        first_ready_control=boundary.first_ready_control,
        good=good,
        ready=ready,
        distance_xy_m=distance,
        relative_speed_xy_mps=relative_speed,
        ball_height_m=float(q[38]),
        forward_m=front,
        root_height_m=float(q[2]),
        upright=upright,
    )
