"""Chronological reward redistribution for the existing finite contact task.

This offline training adapter preserves the original undiscounted terminal
score, not its old constant-per-frame advantage estimator. A caller must bind
and independently audit the full physical trace. Labels are never actor inputs
or proof of action causality, gameplay improvement, or policy authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.online_motor_actor_critic import terminal_return

COMPONENTS = ("base", "quality", "dirty", "out", "unsafe")


@dataclass(frozen=True)
class TemporalContactRewards:
    rewards: np.ndarray[Any, Any]
    component_rewards: np.ndarray[Any, Any]
    first_contact_frame: int | None
    first_dirty_frame: int | None
    first_out_frame: int | None
    first_unsafe_frame: int | None
    evaluation_frame: int | None
    undiscounted_return: float


def _first(mask: np.ndarray[Any, Any]) -> int | None:
    ids = np.flatnonzero(mask)
    return int(ids[0]) if len(ids) else None


def redistribute_contact_rewards(
    trace: dict[str, Any], audited_outcome: dict[str, Any]
) -> TemporalContactRewards:
    """Use post-action measurements from all 300 frames; do not omit failures.

    Progress is provisional until the unchanged first-contact+60 horizon.
    Later dirty contact or boundary crossings revoke provisional quality;
    later instability overrides the base reward exactly as the old scorer.
    No-contact penalties are resolved at the declared finite-task endpoint.
    This task ends at frame 299; this helper does not declare arbitrary
    continuing-control/time-limit rollouts to be true terminal states.
    """
    shapes = {
        "force_n": (300, 1, 6),
        "pelvis_z_per_substep_m": (300, 1, 10),
        "ball_position_after_step_m": (300, 1, 3),
    }
    if any(key not in trace for key in shapes):
        raise ValueError("complete measured contact, pelvis and ball histories required")
    arrays = {key: np.asarray(trace[key]) for key in shapes}
    if any(
        value.shape != shapes[key] or value.dtype.kind != "f" or not np.isfinite(value).all()
        for key, value in arrays.items()
    ):
        raise ValueError("finite full float physical histories required")
    force = arrays["force_n"][:, 0]
    pelvis = arrays["pelvis_z_per_substep_m"][:, 0]
    ball = arrays["ball_position_after_step_m"][:, 0]
    if np.any(force < 0):
        raise ValueError("measured contact force norms cannot be negative")
    contacts = force > 1
    first = _first(np.any(contacts, axis=1))
    dirty = _first(np.any(contacts[:, 2:], axis=1))
    out = _first(np.abs(ball[:, 1]) > 4)
    unsafe = _first(np.min(pelvis, axis=1) < 0.65)
    evaluation = first + 60 if first is not None and first + 60 < 300 else None
    seen = np.maximum.accumulate(contacts, axis=0)
    heights = np.minimum.accumulate(np.min(pelvis, axis=1))
    excursions = np.maximum.accumulate(np.abs(ball[:, 1]))
    levels = np.zeros((300, len(COMPONENTS)), dtype=np.float64)
    for frame in range(300):
        active_first = first if first is not None and frame >= first else None
        bodies = np.flatnonzero(seen[frame]).tolist()
        clean = active_first is not None and not np.any(seen[frame, 2:])
        forward = lateral = None
        if active_first is not None:
            horizon = min(frame, active_first + 60)
            delta = ball[horizon] - ball[active_first]
            forward, lateral = float(delta[0]), float(delta[1])
        measured = dict(
            minimum_pelvis_z_m=float(heights[frame]),
            contact_body_indices=bodies,
            first_contact_frame=active_first,
            forward_60_m=forward,
            lateral_60_m=lateral,
            max_lateral_excursion_m=float(excursions[frame]),
        )
        base = (
            0.0
            if active_first is None and frame < 299 and heights[frame] >= 0.65
            else first_touch_reward(measured)
        )
        quality = bool(
            clean
            and evaluation is not None
            and frame >= evaluation
            and forward is not None
            and lateral is not None
            and forward >= 1
            and abs(lateral) / max(forward, 0.01) <= 0.3
            and excursions[frame] <= 4
        )
        levels[frame] = (
            base,
            10 * quality,
            -8 * (np.any(seen[frame, 2:]) or (frame == 299 and active_first is None)),
            -20 * (excursions[frame] > 4),
            -100 * (heights[frame] < 0.65),
        )
    clean = first is not None and dirty is None
    measured_final = dict(
        first_contact_frame=first,
        contact_body_indices=np.flatnonzero(seen[-1]).tolist(),
        clean_foot_only=clean,
        minimum_pelvis_z_m=float(heights[-1]),
        maximum_lateral_excursion_m=float(excursions[-1]),
        high_quality=bool(levels[-1, 1]),
        safety_passed=bool(heights[-1] >= 0.65),
        **post_contact_displacement(ball, first),
    )
    measured_final["reward"] = first_touch_reward(
        {**measured_final, "max_lateral_excursion_m": measured_final["maximum_lateral_excursion_m"]}
    )
    for key, expected in measured_final.items():
        if key not in audited_outcome:
            raise ValueError(f"missing independently audited outcome: {key}")
        actual = audited_outcome[key]
        if type(expected) in (bool, int, list, type(None)):
            if (
                type(actual) is not type(expected)
                or actual != expected
                or (type(expected) is list and any(type(body) is not int for body in actual))
            ):
                raise ValueError(f"audited outcome differs from measured history: {key}")
        elif (
            type(actual) not in (float, int)
            or not np.isfinite(actual)
            or not np.isclose(actual, expected, atol=1e-8, rtol=0)
        ):
            raise ValueError(f"audited outcome differs from measured history: {key}")
    expected_return = terminal_return(audited_outcome)
    components = np.diff(levels, axis=0, prepend=np.zeros((1, len(COMPONENTS))))
    rewards = np.sum(components, axis=1)
    if not np.isfinite(rewards).all() or not np.isclose(
        rewards.sum(), expected_return, atol=1e-8, rtol=0
    ):
        raise ValueError("chronological rewards must retain the unchanged full terminal return")
    rewards.flags.writeable = False
    components.flags.writeable = False
    return TemporalContactRewards(
        rewards, components, first, dirty, out, unsafe, evaluation, expected_return
    )
