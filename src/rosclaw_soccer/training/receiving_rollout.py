"""Contact-derived credit for an admitted receiving window, not match promotion."""

from __future__ import annotations

from typing import Any

import numpy as np


def receiving_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    required_frames: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        tuple(sorted(set(agent_ids))) != agent_ids
        or agent_id not in agent_ids
        or type(start) is not int
        or type(frames) is not int
        or type(required_frames) is not int
        or not 100 <= required_frames <= 250
        or start < 1
        or not 1 <= frames <= required_frames
    ):
        raise ValueError("explicit roster and admitted receiving window required")
    time = np.asarray(trace["time"])
    n = len(time)
    if (
        time.ndim != 1
        or start + frames > n
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("finite monotonic physical receiving time required")
    if not np.allclose(np.diff(time[max(0, start - 1) : start + frames]), 0.02, atol=1e-8, rtol=0):
        raise ValueError("receiving credit requires actual 50 Hz control frames")
    index = agent_ids.index(agent_id)
    key = agent_id.replace(".", "_")

    def column(name: str, shape: tuple[int, ...]) -> np.ndarray:
        values = np.asarray(trace[name])
        if (
            values.shape != shape
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite aligned receiving evidence required")
        return values

    ball = column("ball_pose", (n, 7))[:, :3]
    speed = np.linalg.norm(column("ball_velocity", (n, 6))[:, :3], axis=1)
    distance = np.minimum(
        *[
            np.linalg.norm(column(key + suffix, (n, 3)) - ball, axis=1)
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
    )
    contact = column("ball_contact_agent_code", (n,))
    effector = column("ball_contact_effector_code", (n,))
    force = column("ball_contact_force_n", (n,))
    nonfoot = column("ball_nonfoot_contact_agent_code", (n,))
    nf_force = column("ball_nonfoot_contact_force_n", (n,))
    if (
        not np.isin(contact, np.arange(len(agent_ids) + 1)).all()
        or not np.isin(nonfoot, np.arange(len(agent_ids) + 1)).all()
        or not np.isin(effector, np.arange(5)).all()
        or np.any(force < 0)
        or np.any(nf_force < 0)
    ):
        raise ValueError("physical receiving contact codes or forces invalid")
    body = column(key + "_pelvis_pose", (n, 7))
    margin = column(key + "_joint_safety_margin_rad", (n, 29)).min(axis=1)
    collision = column("robot_robot_contact_count", (n,))
    if np.any(collision < 0) or not np.allclose(
        np.linalg.norm(body[:, 3:7], axis=1), 1, atol=1e-4, rtol=0
    ):
        raise ValueError("measured normalized body pose and nonnegative collisions required")
    window = slice(start, start + frames)
    foot = ((contact == index + 1) & np.isin(effector, [1, 2]) & (force > 0))[window]
    forbidden = (
        ((nonfoot > 0) & (nf_force > 0)) | ((contact > 0) & (contact != index + 1) & (force > 0))
    )[window]
    upright = 1 - 2 * (body[:, 4] ** 2 + body[:, 5] ** 2)
    safe = ((body[:, 2] >= 0.55) & (upright >= np.cos(0.8)) & (margin >= 0) & (collision == 0))[
        window
    ]
    close_slow = ((distance <= 0.35) & (speed <= 0.35) & (ball[:, 2] <= 0.20))[window]
    touched = np.flatnonzero(foot)
    first = None if not len(touched) else int(touched[0])
    # Distances alone never label reception. Require measured foot contact,
    # an uninterrupted post-contact window and a slow near-foot stable tail.
    controlled = bool(
        first is not None
        and frames == required_frames
        and safe.all()
        and time[start + frames - 1] - time[start + first] >= 0.5 - 1e-8
        and not forbidden[first:].any()
        and close_slow[-10:].all()
    )
    reward = np.zeros(frames, dtype=np.float32)
    seen = False
    for i in range(frames):
        frame = start + i
        if not seen:
            reward[i] += 2 * float(np.clip(distance[frame - 1] - distance[frame], -0.1, 0.1))
        if foot[i] and not forbidden[i]:
            reward[i] += 2 * float(np.clip(speed[frame - 1] - speed[frame], -0.2, 0.2))
            if not seen:
                reward[i] += 2
            seen = True
        if seen and close_slow[i] and safe[i] and not forbidden[i]:
            reward[i] += 0.04
        if forbidden[i] or not safe[i]:
            reward[i] -= 0.1
    reward[-1] += 10 if controlled else -1 if first is not None else -2
    return reward, {
        "schema": (
            "soccer.receiving_window.v1"
            if required_frames == 100
            else "soccer.receiving_window.duration.v2"
        ),
        **({"required_frames": required_frames} if required_frames != 100 else {}),
        "agent_id": agent_id,
        "frames": frames,
        "first_foot_contact_sec": None if first is None else float(time[start + first]),
        "controlled_reception": controlled,
        "minimum_foot_distance_m": float(distance[window].min()),
        "body_and_collision_safe": bool(safe.all()),
        "terminal_ball_speed_mps": float(speed[start + frames - 1]),
        "shaped_return": float(reward.sum()),
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
    }


def explain_receiving_window(
    trace: dict[str, Any],
    *,
    agent_ids: tuple[str, ...],
    agent_id: str,
    start: int,
    frames: int,
    required_frames: int = 100,
) -> dict[str, Any]:
    """Explain every failed capture criterion without changing legacy scores.

    This is a read-only diagnostic, not a task receipt or policy activation.
    Validation and the authoritative decision remain in ``receiving_window``.
    """
    _, outcome = receiving_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=start,
        frames=frames,
        required_frames=required_frames,
    )
    window = slice(start, start + frames)
    key = agent_id.replace(".", "_")
    index = agent_ids.index(agent_id)
    time = np.asarray(trace["time"])[window]
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
    ) | ((contact > 0) & (contact != index + 1) & (force > 0))
    first_time = outcome["first_foot_contact_sec"]
    post_contact = np.zeros(frames, dtype=bool) if first_time is None else time >= first_time
    observation_sec = None if first_time is None else float(time[-1] - first_time)
    tail = slice(-10, None)
    criteria = {
        "measured_own_foot_contact": first_time is not None,
        "complete_window": frames == required_frames,
        "body_and_collision_safe": outcome["body_and_collision_safe"],
        "half_second_after_contact": observation_sec is not None and observation_sec >= 0.5 - 1e-8,
        "no_forbidden_contact_after_touch": not bool(forbidden[post_contact].any()),
        "near_foot_tail": bool((distance[tail] <= 0.35).all()),
        "slow_ball_tail": bool((speed[tail] <= 0.35).all()),
        "low_ball_tail": bool((ball[tail, 2] <= 0.20).all()),
    }
    if all(criteria.values()) != outcome["controlled_reception"]:
        raise ValueError("receiving diagnostic disagrees with authoritative score")
    return {
        "schema": "soccer.receiving_window.explanation.v1",
        "agent_id": agent_id,
        "controlled_reception": outcome["controlled_reception"],
        "criteria": criteria,
        "failed_criteria": [name for name, passed in criteria.items() if not passed],
        "post_contact_observation_sec": observation_sec,
        "forbidden_post_contact_frames": int(np.count_nonzero(forbidden & post_contact)),
        "tail_maximum_foot_distance_m": float(distance[tail].max()),
        "tail_maximum_ball_speed_mps": float(speed[tail].max()),
        "tail_maximum_ball_height_m": float(ball[tail, 2].max()),
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
    }
