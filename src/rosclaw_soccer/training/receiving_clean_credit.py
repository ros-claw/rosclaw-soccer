"""Opt-in credit aligned with irreversible capture failures; no new success gate."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.training.receiving_capture_credit import capture_retention_window


def clean_capture_credit_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    required_frames: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Do not resume hold bonuses after a disqualifying touch or body failure.

    The existing physical scorer validates every input and owns success. This
    explicit research objective adds terminal costs for missing contact,
    irreversible failure, insufficient observation, and tail-distance/speed
    excess. It changes no actuator, geometry, task threshold, or admission.
    """
    reward, outcome = capture_retention_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=start,
        frames=frames,
        required_frames=required_frames,
    )
    reward = reward.copy()
    window = slice(start, start + frames)
    key = agent_id.replace(".", "_")
    ball = np.asarray(trace["ball_pose"])[window, :3]
    speed = np.linalg.norm(np.asarray(trace["ball_velocity"])[window, :3], axis=1)
    distance = np.minimum(
        *[
            np.linalg.norm(np.asarray(trace[key + suffix])[window] - ball, axis=1)
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
    )
    contact = np.asarray(trace["ball_contact_agent_code"])[window]
    force = np.asarray(trace["ball_contact_force_n"])[window]
    forbidden = (
        (np.asarray(trace["ball_nonfoot_contact_agent_code"])[window] > 0)
        & (np.asarray(trace["ball_nonfoot_contact_force_n"])[window] > 0)
    ) | ((contact > 0) & (contact != agent_ids.index(agent_id) + 1) & (force > 0))
    body = np.asarray(trace[key + "_pelvis_pose"])[window]
    safe = (
        (body[:, 2] >= 0.55)
        & (1 - 2 * (body[:, 4] ** 2 + body[:, 5] ** 2) >= np.cos(0.8))
        & (np.asarray(trace[key + "_joint_safety_margin_rad"])[window].min(axis=1) >= 0)
        & (np.asarray(trace["robot_robot_contact_count"])[window] == 0)
    )
    first_time = outcome["first_foot_contact_sec"]
    times = np.asarray(trace["time"])[window]
    seen = np.zeros(frames, dtype=bool) if first_time is None else times >= first_time
    invalid = np.maximum.accumulate(~safe | (forbidden & seen))
    close_slow = (distance <= 0.35) & (speed <= 0.35) & (ball[:, 2] <= 0.20)
    # Remove only hold bonuses that the original scorer actually awarded.
    removed = invalid & seen & close_slow & safe & ~forbidden
    reward[removed] -= np.float32(0.04)
    missing_cost = 8.0 if first_time is None else 0.0
    irreversible_cost = 8.0 if bool(invalid.any()) else 0.0
    observation_cost = (
        0.0 if first_time is None else 4.0 * max(0.0, 1.0 - float(times[-1] - first_time) / 0.5)
    )
    distance_cost = 4.0 * float(np.clip(distance[-10:].max() / 0.35 - 1.0, 0.0, 4.0))
    speed_cost = 4.0 * float(np.clip(speed[-10:].max() / 0.35 - 1.0, 0.0, 4.0))
    terminal_cost = missing_cost + irreversible_cost + observation_cost + distance_cost + speed_cost
    reward[-1] -= np.float32(terminal_cost)
    return reward, {
        **outcome,
        "schema": "soccer.receiving_clean_credit.v1",
        "base_shaped_return": outcome["shaped_return"],
        "removed_ineligible_hold_bonus_frames": int(removed.sum()),
        "irreversible_eligibility_lost": bool(invalid.any()),
        "missing_contact_cost": missing_cost,
        "irreversible_failure_cost": irreversible_cost,
        "post_contact_observation_cost": observation_cost,
        "tail_distance_excess_cost": distance_cost,
        "tail_speed_excess_cost": speed_cost,
        "terminal_criterion_cost": terminal_cost,
        "shaped_return": float(reward.sum()),
    }
