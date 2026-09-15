"""Measured option/contact chronology; never goal adjudication or promotion.

Concurrent 50 Hz contacts have unknown substep ordering. Report that uncertainty
instead of attributing a rebound to the shooter's foot alone. Body contacts may
be legal football; their presence is not a claim of a rule violation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def diagnose_shot_options(
    trace: dict[str, Any], agent_ids: tuple[str, ...], *, goal_planes_x_m: tuple[float, float]
) -> list[dict[str, Any]]:
    time = np.asarray(trace["time"])
    count = len(time)
    if (
        time.ndim != 1
        or not count
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or len(set(agent_ids)) != len(agent_ids)
        or not agent_ids
        or len(goal_planes_x_m) != 2
        or not np.isfinite(goal_planes_x_m).all()
        or not goal_planes_x_m[0] < goal_planes_x_m[1]
    ):
        raise ValueError("ordered finite option feedback contract required")

    def column(key: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(trace[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("finite aligned option feedback columns required")
        return value

    option = column("option_agent_code", (count,))
    contact = column("ball_contact_agent_code", (count,))
    effector = column("ball_contact_effector_code", (count,))
    force = column("ball_contact_force_n", (count,))
    nonfoot = column("ball_nonfoot_contact_agent_code", (count,))
    nonfoot_force = column("ball_nonfoot_contact_force_n", (count,))
    targets = column("option_target_position_m", (count, 3))
    ball = np.asarray(trace["ball_pose"])
    if ball.ndim != 2 or ball.shape[0] != count or ball.shape[1] < 3 or not np.isfinite(ball).all():
        raise ValueError("finite aligned ball poses required")
    for codes in (option, contact, nonfoot):
        if not np.isin(codes, np.arange(len(agent_ids) + 1)).all():
            raise ValueError("unknown physical agent code")
    if np.any(force < 0) or np.any(nonfoot_force < 0):
        raise ValueError("normal contact force cannot be negative")
    results = []
    starts = np.flatnonzero((option > 0) & (option != np.r_[0, option[:-1]]))
    for start_value in starts:
        start = int(start_value)
        code = int(option[start])
        actor = agent_ids[code - 1]
        different = np.flatnonzero(option[start:] != code)
        stop = start + int(different[0]) if len(different) else count
        target = targets[start].copy()
        # Do not interpret a receiver waypoint as a goal-directed shot.
        if not any(abs(float(target[0]) - x) <= 1e-6 for x in goal_planes_x_m):
            continue
        if not np.array_equal(targets[start:stop], np.broadcast_to(target, (stop - start, 3))):
            raise ValueError("motor task target changed inside an option")
        feet = (
            np.flatnonzero(
                (contact[start:stop] == code)
                & np.isin(effector[start:stop], [1, 2])
                & (force[start:stop] > 0)
            )
            + start
        )
        first = None if not len(feet) else int(feet[0])
        end = min(count, int(np.searchsorted(time, time[stop - 1] + 2.0, side="right")))
        bodies = np.flatnonzero((nonfoot[start:end] > 0) & (nonfoot_force[start:end] > 0)) + start
        contacts = []
        for frame in bodies:
            other = agent_ids[int(nonfoot[frame]) - 1]
            contacts.append(
                {
                    "agent_id": other,
                    "time_sec": float(time[frame]),
                    "normal_force_n": float(nonfoot_force[frame]),
                    "relative_to_first_foot": "NO_FOOT_OBSERVED"
                    if first is None
                    else "BEFORE"
                    if frame < first
                    else "SAME_CONTROL_FRAME_ORDER_UNKNOWN"
                    if frame == first
                    else "AFTER",
                    "same_agent": other == actor,
                    "same_team": other.split(".", 1)[0] == actor.split(".", 1)[0],
                }
            )
        crossing = None
        if first is not None:
            direction = float(np.sign(target[0] - ball[first, 0]))
            for frame in range(first + 1, end):
                before, after = ball[frame - 1, :3], ball[frame, :3]
                if direction * (before[0] - target[0]) < 0 <= direction * (after[0] - target[0]):
                    fraction = float((target[0] - before[0]) / (after[0] - before[0]))
                    point = before + fraction * (after - before)
                    interrupted = bool(
                        np.any(
                            (nonfoot[first : frame + 1] > 0)
                            & (nonfoot_force[first : frame + 1] > 0)
                        )
                        or np.any(
                            (contact[first : frame + 1] > 0)
                            & (contact[first : frame + 1] != code)
                            & (force[first : frame + 1] > 0)
                        )
                    )
                    crossing = {
                        "position_m": point.tolist(),
                        "time_sec": float(
                            time[frame - 1] + fraction * (time[frame] - time[frame - 1])
                        ),
                        "target_error_m": float(np.linalg.norm(point[1:] - target[1:])),
                        "intervening_or_concurrent_contact": interrupted,
                        "claim": "BALL_CENTRE_PLANE_CROSSING_NOT_GOAL_ADJUDICATION",
                    }
                    break
        results.append(
            {
                "agent_id": actor,
                "option_start_sec": float(time[start]),
                "option_last_sec": float(time[stop - 1]),
                "task_target_m": target.tolist(),
                "first_foot_contact_sec": None if first is None else float(time[first]),
                "body_contacts": contacts,
                "observed_centre_plane_crossing": crossing,
                "promotion_eligible": False,
                "activation_ceiling": "SIM_ONLY",
            }
        )
    return results
