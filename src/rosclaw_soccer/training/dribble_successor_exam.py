"""Value-only successor exam over synchronous 50 Hz physical observations.

The caller authenticates artifacts, agent identity, geometry and body safety.
This module validates continuity and contact history, not their provenance.
Passing is a SIM_ONLY probe result, never READY, learned skill or promotion.
"""

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DribbleSample:
    frame: int
    time_sec: float
    ball_position_m: tuple[float, float, float]
    ball_velocity_mps: tuple[float, float, float]
    foot_positions_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    last_own_contact_sec: float | None
    last_interruption_sec: float | None


@dataclass(frozen=True)
class DribbleSuccessorResult:
    passed: bool
    reasons: tuple[str, ...]
    goalward_progress_m: float
    maximum_foot_distance_m: float
    terminal_speed_mps: float
    successor_ready_verified: Literal[False] = field(default=False, init=False)
    training_authorized: Literal[False] = field(default=False, init=False)
    activation_ceiling: Literal["SIM_ONLY"] = field(default="SIM_ONLY", init=False)


def _finite(value: object) -> bool:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        return False
    return abs(value) <= 1e6 and math.isfinite(value)


def evaluate_dribble_successor(
    samples: tuple[DribbleSample, ...],
    *,
    direction: Literal[-1, 1],
    episode_safe: bool,
    profile: Literal["one_second", "two_second"],
) -> DribbleSuccessorResult:
    """Require admission sample plus exactly 50/100 subsequent ordered samples.

    Admission itself must be separately proven by ReceivingControlWitness. Invalid
    or incomplete evidence raises ValueError; physically unsuccessful valid evidence
    returns passed=False. Thresholds are fixed profiles, not tunable call arguments.
    """
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("explicit attacking direction required")
    if type(episode_safe) is not bool or profile not in ("one_second", "two_second"):
        raise ValueError("strict safety and known exam profile required")
    count = 50 if profile == "one_second" else 100
    if type(samples) is not tuple or len(samples) != count + 1:
        raise ValueError("complete fixed-rate admission and horizon required")
    previous: DribbleSample | None = None
    for sample in samples:
        if not isinstance(sample, DribbleSample):
            raise ValueError("typed synchronous samples required")
        if type(sample.frame) is not int or sample.frame < 0 or not _finite(sample.time_sec):
            raise ValueError("finite nonnegative frame and clock required")
        if sample.time_sec < 0:
            raise ValueError("negative clock")
        vectors = (sample.ball_position_m, sample.ball_velocity_mps)
        if type(sample.foot_positions_m) is not tuple or len(sample.foot_positions_m) != 2:
            raise ValueError("two synchronous feet required")
        for vector in (*vectors, *sample.foot_positions_m):
            if type(vector) is not tuple or len(vector) != 3 or not all(map(_finite, vector)):
                raise ValueError("finite three-dimensional geometry required")
        for name in ("last_own_contact_sec", "last_interruption_sec"):
            value = getattr(sample, name)
            if value is not None and (
                not _finite(value) or not 0 <= value <= sample.time_sec + 1e-9
            ):
                raise ValueError("contact history cannot be nonfinite or future")
            old = getattr(previous, name) if previous is not None else None
            if old is not None and (value is None or value < old):
                raise ValueError("contact history cannot disappear or regress")
        if previous is not None and (
            sample.frame != previous.frame + 1
            or not math.isclose(sample.time_sec - previous.time_sec, 0.02, abs_tol=1e-9)
        ):
            raise ValueError("observation gap, duplicate or clock mismatch")
        previous = sample
    start = samples[0]
    end = samples[-1]
    measured = samples[1:]
    progress = direction * (end.ball_position_m[0] - start.ball_position_m[0])
    distance = max(
        min(math.dist(s.ball_position_m, foot) for foot in s.foot_positions_m) for s in measured
    )
    speed = math.sqrt(sum(v * v for v in end.ball_velocity_mps))
    reasons: list[str] = []
    if not episode_safe:
        reasons.append("unsafe_episode")
    if progress < (0.2 if count == 50 else 0.3):
        reasons.append("insufficient_progress")
    if distance > 0.55:
        reasons.append("ball_out_of_reach")
    if any(not 0 <= s.ball_position_m[2] <= 0.2 for s in measured):
        reasons.append("ball_height")
    if speed > 0.8:
        reasons.append("terminal_ball_speed")
    if any(
        s.last_interruption_sec is not None and s.last_interruption_sec > start.time_sec + 1e-9
        for s in measured
    ):
        reasons.append("contact_interruption")
    touch_boundary = start.time_sec + (1.0 if count == 100 else 0.0)
    if not any(
        s.last_own_contact_sec is not None and s.last_own_contact_sec > touch_boundary + 1e-9
        for s in measured
    ):
        reasons.append("no_new_foot_touch" if count == 50 else "no_second_half_foot_touch")
    return DribbleSuccessorResult(not reasons, tuple(reasons), progress, distance, speed)
