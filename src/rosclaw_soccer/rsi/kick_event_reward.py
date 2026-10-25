"""Causal finite-episode motor rewards, never a replacement for the H300 exam.

Reward at control t uses only that control's actual ten native measurements and
earlier observations. A ball-flight bonus is not known until first_foot+60.
Use undiscounted finite returns: delaying a safety cost must not make it cheap.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.sim.contracts import hash_json

COMPONENTS = (
    "first_foot",
    "observed_quality",
    "nonfoot",
    "body_failure",
    "out_of_play",
    "deadline",
)


def event_reward_contract() -> dict[str, Any]:
    contract = dict(
        schema="soccer.g1.causal_kick_event_reward.v1",
        control_hz=50,
        native_samples_per_control=10,
        formal_task_controls=300,
        physical_episode_max_controls=360,
        post_contact_controls=60,
        contact_force_threshold_n=1.0,
        minimum_body_height_m=0.65,
        minimum_body_upright=0.9,
        first_foot_reward=1.0,
        quality_bonus="observed prefix first_touch_reward minus first_foot_reward",
        maximum_total_positive_reward=4.0,
        first_nonfoot_cost=10.0,
        first_body_failure_cost=10.0,
        first_out_of_play_cost=4.0,
        no_foot_deadline_cost=3.0,
        censored_contact_deadline_cost=2.0,
        nonfoot_and_out_of_play_window="controls0..299",
        body_failure_window="all actual controls up to359",
        discount=1.0,
        bootstrap_after_episode=False,
        worst_unsafe_total_return_upper_bound=-6.0,
        future_measurements_never_change_prior_rewards=True,
        new_training_objective_not_legacy_terminal_reward=True,
        formal_exam_unchanged=True,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(contract, contract_hash=hash_json(contract))


def kick_event_rewards(
    *, ball_after: np.ndarray, pelvis_native: np.ndarray, force_native: np.ndarray
) -> dict[str, Any]:
    """Accept an actual prefix, never pad it to infer unobserved deadlines."""
    n = len(ball_after)
    if not 1 <= n <= 360 or any(
        x.shape != shape or x.dtype != np.float64 or not np.isfinite(x).all()
        for x, shape in (
            (ball_after, (n, 3)),
            (pelvis_native, (n, 10, 7)),
            (force_native, (n, 10, 3)),
        )
    ):
        raise ValueError("finite float64 actual control prefixes with ten native samples required")
    if (force_native < 0).any() or not np.allclose(
        np.linalg.norm(pelvis_native[:, :, 3:7], axis=-1), 1.0, atol=1e-5, rtol=0
    ):
        raise ValueError(
            "nonnegative measured forces and actual normalized body quaternions required"
        )
    components = np.zeros((n, len(COMPONENTS)), np.float64)
    first_foot = nonfoot = body_failure = out_of_play = quality_observed = quality_reward = None
    minimum_height, excursion = float("inf"), 0.0
    initial_ball = None
    first_bodies: list[int] = []
    for t in range(n):
        heights = pelvis_native[t, :, 2]
        quat = pelvis_native[t, :, 3:7]
        upright = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        minimum_height = min(minimum_height, float(heights.min()))
        if body_failure is None and ((heights < 0.65) | (upright < 0.9)).any():
            body_failure = t
            components[t, 3] = -10.0
        if t >= 300:
            continue  # Body recovery still matters, but the football task is closed.
        force = force_native[t].max(axis=0)
        excursion = max(excursion, float(abs(ball_after[t, 1])))
        if nonfoot is None and force[2] > 1:
            nonfoot = t
            components[t, 2] = -10.0
        if out_of_play is None and excursion > 4:
            out_of_play = t
            components[t, 4] = -4.0
        if first_foot is None and (force[:2] > 1).any():
            first_foot = t
            initial_ball = ball_after[t].copy()
            first_bodies = np.flatnonzero(force > 1).tolist()
            components[t, 0] = 1.0
        if first_foot is not None and t == first_foot + 60:
            quality_observed = t
            if nonfoot is None and body_failure is None and out_of_play is None:
                assert initial_ball is not None
                displacement = ball_after[t] - initial_ball
                score = first_touch_reward(
                    dict(
                        first_contact_frame=first_foot,
                        contact_body_indices=first_bodies,
                        minimum_pelvis_z_m=minimum_height,
                        max_lateral_excursion_m=excursion,
                        forward_60_m=float(displacement[0]),
                        lateral_60_m=float(displacement[1]),
                    )
                )
                components[t, 1] = score - 1.0
                quality_reward = t
        if t == 299:
            if first_foot is None:
                components[t, 5] = -3.0
            elif quality_observed is None:
                components[t, 5] = -2.0  # With the earlier +1, censored total is -1.
    rewards = components.sum(axis=1).astype(np.float32)
    if components.clip(min=0).sum() > 4 + 1e-10:
        raise ValueError("causal positive reward budget exceeded")
    if (nonfoot is not None or body_failure is not None) and rewards.sum(
        dtype=np.float64
    ) > -6 + 1e-5:
        raise ValueError("unsafe behavior cannot purchase a positive football return")
    return dict(
        rewards=rewards,
        components=components,
        component_names=COMPONENTS,
        first_foot_control=first_foot,
        first_nonfoot_control=nonfoot,
        first_body_failure_control=body_failure,
        first_out_of_play_control=out_of_play,
        quality_observed_control=quality_observed,
        quality_reward_control=quality_reward,
        actual_controls=n,
        task_deadline_observed=n >= 300,
        discount=1.0,
        formal_exam_unchanged=True,
    )
