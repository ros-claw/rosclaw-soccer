"""Build causal pre-contact learner inputs from audited Isaac trajectories.

All features precede the first measured ball/body impact.  This prevents a
learner from using the contact result itself as a shortcut observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.ball_contact_evidence import classify_ball_body_contacts

BODY_ORDER = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
)


@dataclass(frozen=True)
class PrecontactSamples:
    frame_indices: NDArray[np.int64]
    features: NDArray[np.float64]
    clean_foot_only: bool
    first_contact_microstep: int | None


def extract_precontact_samples(
    arrays: Mapping[str, NDArray[np.float64]],
    *,
    body_names: tuple[str, ...],
    window_frames: int = 20,
) -> PrecontactSamples:
    """Return observed ball/foot/knee geometry strictly before impact.

    Feature columns: for left and right foot, ball-relative position and
    velocity (12); for both knees, ball-relative position (6).  The sample
    label comes from all physics microsteps of the *whole* episode.
    """
    if body_names != BODY_ORDER or not 1 <= window_frames <= 100:
        raise ValueError("qualified body order and bounded sample window required")
    ball = np.asarray(arrays["ball_observation_position_m"], dtype=np.float64)
    ball_velocity = np.asarray(arrays["ball_observation_velocity_m_s"], dtype=np.float64)
    position = np.asarray(arrays["contact_body_position_m"], dtype=np.float64)
    velocity = np.asarray(arrays["contact_body_velocity_m_s"], dtype=np.float64)
    force = np.asarray(arrays["ball_body_contact_force_micro_n"], dtype=np.float64)
    frames = len(ball)
    if (
        frames < 2
        or ball.shape != (frames, 3)
        or ball_velocity.shape != (frames, 3)
        or position.shape != (frames, 1, 4, 3)
        or velocity.shape != position.shape
        or force.shape[:2] != (frames, 10)
        or force.shape[2] != 6
        or not all(
            np.isfinite(value).all() for value in (ball, ball_velocity, position, velocity, force)
        )
    ):
        raise ValueError("complete finite Isaac contact trajectory required")
    summary = classify_ball_body_contacts(force)
    impacts = (summary.first_foot_microstep, summary.first_nonfoot_microstep)
    first_impact = min((step for step in impacts if step is not None), default=None)
    cutoff = min(frames, first_impact // 10 if first_impact is not None else frames)
    if cutoff < 2:
        raise ValueError("at least two pre-contact frames required")
    indices = np.arange(max(1, cutoff - window_frames), cutoff, dtype=np.int64)
    ball_velocity = ball_velocity[indices]
    foot_position = position[indices, 0, :2]
    foot_velocity = velocity[indices, 0, :2]
    knee_position = position[indices, 0, 2:]
    relative_position = ball[indices, None, :] - foot_position
    relative_velocity = ball_velocity[:, None, :] - foot_velocity
    knee_relative = ball[indices, None, :] - knee_position
    features = np.concatenate(
        (
            relative_position.reshape(len(indices), 6),
            relative_velocity.reshape(len(indices), 6),
            knee_relative.reshape(len(indices), 6),
        ),
        axis=1,
    )
    if features.shape != (len(indices), 18) or not np.isfinite(features).all():
        raise ValueError("invalid pre-contact learner features")
    return PrecontactSamples(indices, features, summary.clean_foot_only, first_impact)
