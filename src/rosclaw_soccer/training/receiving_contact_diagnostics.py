"""Continuous contact diagnostics, not a replacement receiving success gate.

Phase hints are causal descriptive labels, not annotated demonstrations or
policy authority. READY, COM and support are unknown without separate evidence.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_rollout import receiving_window
from rosclaw_soccer.training.returned_ball_learning import require_returned_live_segment


def receiving_contact_diagnostics(
    trace: dict[str, Any], *, agent_ids: tuple[str, ...], agent_id: str
) -> dict[str, Any]:
    """Measure margins, causal finite-difference velocities and control duration.

    Foot positions are recorded body origins, not signed surface distances.
    The first velocity sample is unknown and explicitly masked, never inferred
    from a future frame. A nearest-foot switch does not create a velocity spike.
    """
    require_returned_live_segment(trace)
    time = np.asarray(trace["time"])
    if (
        time.ndim != 1
        or not 2 <= len(time) <= 50_000
        or time.dtype.kind not in "fiu"
        or not np.isfinite(time).all()
        or np.any(time < 0)
        or not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-7)
    ):
        raise ValueError("bounded contiguous measured contact timeline required")
    # Reuse validation for body, joint safety, ball, foot and contact evidence.
    receiving_window(
        trace, agent_ids=agent_ids, agent_id=agent_id, start=1, frames=min(100, len(time) - 1)
    )
    n = len(time)
    code = agent_ids.index(agent_id) + 1
    key = agent_id.replace(".", "_")
    ball = np.asarray(trace["ball_pose"])[:, :3]
    ball_velocity = np.asarray(trace["ball_velocity"])[:, :3]
    feet = np.stack(
        [
            np.asarray(trace[key + suffix])
            for suffix in ("_left_foot_position", "_right_foot_position")
        ],
        axis=1,
    )
    body = np.asarray(trace[key + "_pelvis_pose"])
    distance = np.linalg.norm(feet - ball[:, None, :], axis=2)
    foot_velocity = np.zeros_like(feet, dtype=np.float64)
    pelvis_velocity = np.zeros((n, 3))
    foot_velocity[1:] = np.diff(feet, axis=0) / np.diff(time)[:, None, None]
    pelvis_velocity[1:] = np.diff(body[:, :3], axis=0) / np.diff(time)[:, None]
    velocity_observed = np.arange(n) > 0
    contacts = np.asarray(trace["ball_contact_agent_code"])
    effectors = np.asarray(trace["ball_contact_effector_code"])
    forces = np.asarray(trace["ball_contact_force_n"])
    foot_contact = (contacts == code) & np.isin(effectors, (1, 2)) & (forces > 0)
    forbidden = (
        (np.asarray(trace["ball_nonfoot_contact_agent_code"]) > 0)
        & (np.asarray(trace["ball_nonfoot_contact_force_n"]) > 0)
    ) | ((contacts > 0) & (contacts != code) & (forces > 0))
    upright = 1 - 2 * (body[:, 4] ** 2 + body[:, 5] ** 2)
    safe = (
        (body[:, 2] >= 0.55)
        & (upright >= np.cos(0.8))
        & (np.asarray(trace[key + "_joint_safety_margin_rad"]).min(axis=1) >= 0)
        & (np.asarray(trace["robot_robot_contact_count"]) == 0)
    )
    speed = np.linalg.norm(ball_velocity, axis=1)
    close_slow = (distance.min(axis=1) <= 0.35) & (speed <= 0.35) & (ball[:, 2] <= 0.20)
    selected_feet = distance.argmin(axis=1)
    clean_history = np.zeros(n, dtype=bool)
    contact_age = np.zeros(n)
    control_duration = np.zeros(n)
    phase = np.full(n, "APPROACH", dtype="U16")
    first_clean_contact: float | None = None
    qualified_since: float | None = None
    last_contact_foot: int | None = None
    for i in range(n):
        if forbidden[i] or not safe[i]:
            first_clean_contact = qualified_since = None
            last_contact_foot = None
        elif foot_contact[i]:
            if first_clean_contact is None:
                first_clean_contact = float(time[i])
            last_contact_foot = int(effectors[i]) - 1
        if last_contact_foot is not None:
            selected_feet[i] = last_contact_foot
        clean_history[i] = first_clean_contact is not None
        if first_clean_contact is not None:
            contact_age[i] = float(time[i]) - first_clean_contact
        if first_clean_contact is not None and close_slow[i] and safe[i] and not forbidden[i]:
            if qualified_since is None:
                qualified_since = float(time[i])
            control_duration[i] = float(time[i]) - qualified_since
        else:
            qualified_since = None
        if not safe[i]:
            phase[i] = "UNSAFE"
        elif forbidden[i]:
            phase[i] = "INTERRUPTED"
        elif foot_contact[i]:
            phase[i] = "CONTACT"
        elif clean_history[i]:
            phase[i] = (
                "CONTROL"
                if contact_age[i] >= 0.5 - 1e-8 and control_duration[i] >= 0.2 - 1e-8
                else "FOLLOW"
                if speed[i] <= 0.35
                else "ABSORB"
            )
        elif distance[i].min() <= 0.35:
            phase[i] = "PRE_CONTACT"
    selected_velocity = foot_velocity[np.arange(n), selected_feet]
    relative = ball_velocity - selected_velocity
    relative[~velocity_observed] = 0
    if any(
        not np.isfinite(values).all()
        for values in (distance, speed, selected_velocity, relative, pelvis_velocity)
    ):
        raise ValueError("nonfinite derived contact measurement")
    return {
        "schema": "soccer.receiving_contact_diagnostics.v1",
        "time": time.copy(),
        "phase_hint": phase,
        "phase_is_success_certificate": False,
        "selected_foot_code": selected_feet + 1,
        "physical_foot_contact": foot_contact,
        "clean_contact_history": clean_history,
        "contact_age_sec": contact_age,
        "continuous_control_sec": control_duration,
        "distance_margin_m": 0.35 - distance.min(axis=1),
        "velocity_margin_mps": 0.35 - speed,
        "height_margin_m": 0.20 - ball[:, 2],
        "velocity_observed": velocity_observed,
        "foot_velocity_mps": selected_velocity,
        "ball_relative_foot_velocity_mps": relative,
        "pelvis_velocity_mps": pelvis_velocity,
        "velocity_method": "causal_backward_difference_of_recorded_body_origins",
        "support_state": None,
        "center_of_mass": None,
        "successor_readiness": None,
        "promotion_authorized": False,
    }
