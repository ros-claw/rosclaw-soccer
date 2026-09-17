"""Measured, non-overlapping receiving trials inside live team trajectories.

Admission is causal (task and foot distance), never conditioned on a future
touch or success. The existing two-second capture contract stays authoritative.
This is training credit, not a dribbling, team-pass or promotion certificate.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.receiving_capture_credit import capture_retention_window
from rosclaw_soccer.training.returned_ball_learning import require_returned_live_segment


def team_capture_trials(
    trace: dict[str, Any], *, agent_ids: tuple[str, ...]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Admit RECEIVE/PRESS/INTERCEPT within 1 m of a measured foot.

    Once admitted, score the entire next 100 control frames even if the task
    changes: declaring PASS cannot erase a failed capture. Incomplete terminal
    windows are recorded as censored failures, never successful captures. No
    window may span an external throw or dead-ball interval. A new window for
    this player starts only after the previous one ends; touches do not restart
    the clock. Other players are evaluated independently, without ownership
    labels or handshakes substituting for measured contact.
    """
    require_returned_live_segment(trace)
    time = np.asarray(trace["time"])
    if (
        not agent_ids
        or tuple(sorted(set(agent_ids))) != agent_ids
        or time.ndim != 1
        or not len(time)
        or time.dtype.kind not in "fiu"
        or not np.isfinite(time).all()
        or time[0] < 0
        or not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-7)
    ):
        raise ValueError("sorted roster and contiguous 50 Hz timeline required")
    n = len(time)
    ball = np.asarray(trace["ball_pose"])
    if ball.shape != (n, 7) or not np.isfinite(ball).all():
        raise ValueError("aligned finite measured ball pose required")
    reward = np.zeros((n, len(agent_ids)), dtype=np.float64)
    trials: list[dict[str, Any]] = []
    allowed = [
        list(TacticalIntent).index(intent)
        for intent in (TacticalIntent.RECEIVE, TacticalIntent.PRESS, TacticalIntent.INTERCEPT)
    ]
    for column, agent in enumerate(agent_ids):
        key = agent.replace(".", "_")
        intent = np.asarray(trace[key + "_intent_code"])
        feet = [
            np.asarray(trace[key + suffix])
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
        if (
            intent.shape != (n,)
            or intent.dtype.kind not in "iu"
            or not np.isin(intent, range(len(TacticalIntent))).all()
            or any(f.shape != (n, 3) or not np.isfinite(f).all() for f in feet)
        ):
            raise ValueError("measured aligned feet and valid intent codes required")
        distance = np.minimum(*(np.linalg.norm(f - ball[:, :3], axis=1) for f in feet))
        admitted = np.isin(intent, allowed) & (distance <= 1.0)
        # The scorer requires the preceding measured frame; no invented initial
        # difference and no borrowing a frame across an external return.
        start = 1
        while start < n:
            if not admitted[start]:
                start += 1
                continue
            frames = min(100, n - start)
            values, outcome = capture_retention_window(
                trace, agent_ids=agent_ids, agent_id=agent, start=start, frames=frames
            )
            reward[start : start + frames, column] += values
            trials.append(
                {
                    **outcome,
                    "start_frame": start,
                    "start_time_sec": float(time[start]),
                    "censored": frames < 100,
                    "admission": "receiving_task_and_measured_foot_distance_le_1m",
                }
            )
            start += frames
    return reward, trials
