"""Measured ball-task response, not intent labels or match-success certificates."""

from typing import Any

import numpy as np


def ball_engagement_metrics(
    *,
    time: np.ndarray,
    player_xy: np.ndarray,
    ball_xy: np.ndarray,
    eligible: np.ndarray,
    foot_contact: np.ndarray,
) -> dict[str, Any]:
    """Evaluate contiguous 50 Hz epochs in which this player must engage the ball.

    The caller supplies actual task eligibility, excluding dead-ball assistance,
    recovery, support and goalkeeper positioning. Closing speed uses the player's
    own displacement toward the ball, NOT shrinking range caused by ball flight.
    Response requires three consecutive intervals above 0.1 m/s; contact is a
    separately supplied measured foot collision. Missing responses are censored
    (None), never reported as zero latency. This diagnostic grants no promotion.
    """
    if time.ndim != 1 or not len(time):
        raise ValueError("nonempty measured timeline required")
    for a, shape in ((time, time.shape), (player_xy, (len(time), 2)), (ball_xy, (len(time), 2))):
        if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
            raise ValueError("finite aligned clock and positions required")
    if time[0] < 0 or not np.allclose(np.diff(time), 0.02, rtol=0, atol=1e-7):
        raise ValueError("contiguous 50 Hz measurements required")
    for a in (eligible, foot_contact):
        if a.shape != time.shape or a.dtype != np.bool_:
            raise ValueError("explicit aligned boolean task/contact evidence required")
    distance = np.linalg.norm(ball_xy - player_xy, axis=1)
    direction = (ball_xy - player_xy) / np.maximum(distance[:, None], 1e-9)
    velocity = np.zeros_like(player_xy, dtype=float)
    velocity[1:] = np.diff(player_xy, axis=0) / 0.02
    closing = np.sum(velocity * direction, axis=1)
    starts = np.flatnonzero(eligible & ~np.r_[False, eligible[:-1]])
    ends = np.flatnonzero(eligible & ~np.r_[eligible[1:], False]) + 1
    rows = []
    for start, end in zip(starts, ends, strict=True):
        # The interval ending at onset belongs partly to the previous task.
        response = None
        for frame in range(start + 3, end):
            if np.all(closing[frame - 2 : frame + 1] >= 0.1):
                response = float(time[frame] - time[start])
                break
        hits = np.flatnonzero(foot_contact[start:end])
        intervals = closing[start + 1 : end]
        rows.append(
            dict(
                start_sec=float(time[start]),
                duration_sec=float((end - start) * 0.02),
                initial_ball_distance_m=float(distance[start]),
                minimum_ball_distance_m=float(distance[start:end].min()),
                sustained_approach_latency_sec=response,
                first_foot_contact_latency_sec=(float(hits[0] * 0.02) if len(hits) else None),
                foot_contact_frames=int(len(hits)),
                own_toward_ball_distance_m=float(np.maximum(intervals, 0).sum() * 0.02),
                own_away_from_ball_distance_m=float(np.maximum(-intervals, 0).sum() * 0.02),
                measured_intervals=len(intervals),
            )
        )
    return dict(
        opportunities=rows,
        opportunity_count=len(rows),
        responded_count=sum(r["sustained_approach_latency_sec"] is not None for r in rows),
        contacted_count=sum(r["first_foot_contact_latency_sec"] is not None for r in rows),
        match_qualified=False,
        metric_scope="TASK_CONDITIONED_PLAYER_MOTION_AND_MEASURED_FOOT_CONTACT",
    )
