"""Versioned finite-horizon labels; missing ball travel is never fabricated.

This training contract does not modify the legacy exam or its controller.
Artifact callers must separately authenticate raw traces and replay receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.sim.contracts import hash_json


def outcome_contract() -> dict[str, Any]:
    value = dict(
        schema="soccer.rsi.first_touch_outcome.v2",
        horizon_frames=300,
        post_contact_frames=60,
        contact_force_threshold_n=1.0,
        minimum_safe_pelvis_m=0.65,
        unsafe_reward=-10.0,
        no_foot_contact_reward=-3.0,
        censored_contact_reward=-1.0,
        observed_reward="unchanged contextual_first_touch_option.first_touch_reward",
        formal_exam_unchanged=True,
        deadline_is_task_termination=True,
        bootstrap_after_task_deadline=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    return dict(value, contract_hash=hash_json(value))


@dataclass(frozen=True, slots=True)
class FirstTouchOutcome:
    """Labels constructed from complete measured force, ball and pelvis arrays."""

    first_foot_contact_frame: int | None
    event_status: str
    quality_observed: bool
    measurement_status: str
    forward_60_m: float | None
    lateral_60_m: float | None
    task_pass: bool
    failure_reason: str | None
    minimum_pelvis_z_m: float
    maximum_lateral_excursion_m: float
    safe: bool
    training_reward: float

    def to_dict(self) -> dict[str, Any]:
        return dict(
            asdict(self),
            outcome_contract_hash=outcome_contract()["contract_hash"],
            terminated=True,
            truncated=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )


def label_first_touch(trace: dict[str, Any]) -> FirstTouchOutcome:
    """Fixed300-frame, foot-only task. Even unsafe outcomes retain measurements."""
    if type(trace) is not dict:
        raise ValueError("ordinary measured trace required")
    arrays = []
    for name, shape in (
        ("force_n", (300, 1, 6)),
        ("pelvis_z_per_substep_m", (300, 1, 10)),
        ("ball_position_after_step_m", (300, 1, 3)),
    ):
        value = np.asarray(trace.get(name))
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("complete300 finite float64 physical measurements required")
        arrays.append(value[:, 0])
    force, height, ball = arrays
    if np.any(force < 0):
        raise ValueError("contact forces must be nonnegative")
    contacts = np.flatnonzero(np.any(force[:, :2] > 1, axis=1))
    first = int(contacts[0]) if len(contacts) else None
    observed = first is not None and first + 60 < 300
    forward = (
        float(ball[first + 60, 0] - ball[first, 0]) if first is not None and observed else None
    )
    lateral = (
        float(ball[first + 60, 1] - ball[first, 1]) if first is not None and observed else None
    )
    minimum = float(np.min(height))
    excursion = float(np.max(np.abs(ball[:, 1])))
    safe = minimum >= 0.65 and not bool(np.any(force[:, 2:] > 1))
    passed = bool(
        safe
        and forward is not None
        and lateral is not None
        and forward >= 1
        and abs(lateral) / max(forward, 0.01) <= 0.3
        and excursion <= 4
    )
    measurement = (
        "OBSERVED" if observed else ("RIGHT_CENSORED" if first is not None else "NOT_APPLICABLE")
    )
    reason: str | None
    if not safe:
        status, reason, reward = "UNSAFE_CONTACT", "SAFETY_VETO", -10.0
    elif first is None:
        status, reason, reward = "NO_FOOT_CONTACT", "NO_FOOT_CONTACT", -3.0
    elif not observed:
        status, reason, reward = "POST_CONTACT_CENSORED", "POST_CONTACT_WINDOW_INCOMPLETE", -1.0
    else:
        status = "CONTACT_OBSERVED"
        reason = None if passed else "BALL_OUTCOME_FAILED"
        reward = first_touch_reward(
            dict(
                first_contact_frame=first,
                contact_body_indices=np.flatnonzero(force[first] > 1).tolist(),
                minimum_pelvis_z_m=minimum,
                max_lateral_excursion_m=excursion,
                forward_60_m=forward,
                lateral_60_m=lateral,
            )
        )
    return FirstTouchOutcome(
        first,
        status,
        observed,
        measurement,
        forward,
        lateral,
        passed,
        reason,
        minimum,
        excursion,
        safe,
        reward,
    )


def remaining_task_time(frame: int) -> float:
    """Explicit clock for a NEW observation adapter, not a legacy feature change."""
    if type(frame) is not int or not 0 <= frame < 300:
        raise ValueError("task frame must lie in0..299")
    return (299 - frame) / 299
