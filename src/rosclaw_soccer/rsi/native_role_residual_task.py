"""Declared continuing control reward for native role-residual research.

Role placement and real foot/ball interaction, not command-label MSE. Close
recent co-motion is a diagnostic proxy, not adjudicated possession or a goal.
The12sec sampler truncates a continuing task; only body failure terminates it.
"""

import numpy as np

from rosclaw_soccer.rsi.native_team_intent import ROLES, role_target
from rosclaw_soccer.sim.contracts import hash_json


def residual_reward_contract() -> dict[str, object]:
    return dict(
        schema="soccer.native_role_residual_reward.v1",
        control_hz=50,
        progress_gain=3.0,
        progress_clip_m=0.1,
        proximity_match_gain=0.1,
        foot_distance_scale_m=0.3,
        relative_speed_scale_mps=0.5,
        actual_foot_contact_gain=0.2,
        recent_close_comotion_gain=0.5,
        recent_contact_controls=25,
        close_foot_gap_m=0.35,
        close_relative_speed_mps=0.5,
        interplayer_contact_cost=0.2,
        own_body_failure_cost=5.0,
        raw_residual_cost=0.01,
        proximity_and_comotion_are_not_possession_qualification=True,
        duration_is_sampler_truncation_not_task_termination=True,
        active_contact_threshold_newtons=1.0,
    )


def residual_reward_hash() -> str:
    return str(hash_json(residual_reward_contract()))


def native_role_residual_reward(
    *,
    before_roots: np.ndarray,
    after_roots: np.ndarray,
    before_ball: np.ndarray,
    after_ball: np.ndarray,
    foot_gap: np.ndarray,
    foot_relative_speed: np.ndarray,
    contact: np.ndarray,
    contact_age: np.ndarray,
    collision: np.ndarray,
    body_failed: np.ndarray,
    raw_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = (
        before_roots,
        after_roots,
        before_ball,
        after_ball,
        foot_gap,
        foot_relative_speed,
        raw_action,
    )
    if (
        any(not np.isfinite(v).all() for v in numeric)
        or any(v.shape != (8, 2) for v in (before_roots, after_roots))
        or any(v.shape != (2,) for v in (before_ball, after_ball))
        or any(
            v.shape != (8,)
            for v in (foot_gap, foot_relative_speed, contact, contact_age, collision, body_failed)
        )
        or raw_action.shape != (8, 3)
        or any(v.dtype != np.bool_ for v in (contact, collision, body_failed))
        or contact_age.dtype.kind not in "iu"
        or not np.array_equal(contact_age == 0, contact)
        or np.any(contact_age < 0)
        or np.any(foot_gap < 0)
        or np.any(foot_relative_speed < 0)
        or np.any(abs(raw_action) > 1)
    ):
        raise ValueError("finite aligned native8 measurements and actual event masks required")
    distances = []
    for roots, ball in ((before_roots, before_ball), (after_roots, after_ball)):
        distances.append(
            np.array(
                [
                    np.linalg.norm(
                        roots[i] - (ball if role == "finisher" else role_target(team, role, ball))
                    )
                    for i, (team, role) in enumerate((t, r) for t in ("red", "blue") for r in ROLES)
                ]
            )
        )
    close = (contact_age <= 25) & (foot_gap < 0.35) & (foot_relative_speed < 0.5)
    components = np.stack(
        (
            3 * np.clip(distances[0] - distances[1], -0.1, 0.1),
            0.1 * np.exp(-foot_gap / 0.3) * np.exp(-foot_relative_speed / 0.5),
            0.2 * contact,
            0.5 * close,
            -0.2 * collision,
            -5.0 * body_failed,
            -0.01 * np.square(raw_action).mean(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    return components.sum(axis=1), components
