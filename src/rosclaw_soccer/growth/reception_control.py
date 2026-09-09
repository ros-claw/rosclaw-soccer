"""Measured stop-control exam: a foot touch alone is not a trapped pass.

This is not a one-touch relay or shot evaluator, nor a promotion service.
All arrays must describe the same physical post-step clock and player.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReceptionControlConfig:
    observation_sec: float = 0.50
    stable_tail_sec: float = 0.20
    maximum_speed_mps: float = 0.50
    maximum_foot_distance_m: float = 0.35
    minimum_incoming_speed_mps: float = 0.40
    maximum_sample_gap_sec: float = 0.04

    def __post_init__(self) -> None:
        v = asdict(self)
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in v.values()) or not (
            0.3 <= self.observation_sec <= 1.0
            and 0.1 <= self.stable_tail_sec <= self.observation_sec
            and 0.1 <= self.maximum_speed_mps <= 0.6
            and 0.15 <= self.maximum_foot_distance_m <= 0.4
            and 0.1 <= self.minimum_incoming_speed_mps <= 2.0
            and 0.001 <= self.maximum_sample_gap_sec <= 0.05
        ):
            raise ValueError("bounded finite stop-control exam thresholds required")


def measure_reception_control(
    *,
    time: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    receiver_feet: np.ndarray,
    foot_contact_force: np.ndarray,
    forbidden_contact: np.ndarray,
    body_safe: np.ndarray,
    contact_index: int,
    config: ReceptionControlConfig | None = None,
) -> dict[str, Any]:
    if config is not None and not isinstance(config, ReceptionControlConfig):
        raise ValueError("typed stop-control exam configuration required")
    config = config or ReceptionControlConfig()
    time = np.asarray(time, dtype=float)
    n = len(time) if time.ndim == 1 else 0
    ball = np.asarray(ball_position, dtype=float)
    velocity = np.asarray(ball_velocity, dtype=float)
    feet = np.asarray(receiver_feet, dtype=float)
    force = np.asarray(foot_contact_force, dtype=float)
    forbidden = np.asarray(forbidden_contact)
    safe = np.asarray(body_safe)
    if (
        n < 2
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or np.any(np.diff(time) > config.maximum_sample_gap_sec + 1e-9)
        or ball.shape != (n, 3)
        or velocity.shape != (n, 3)
        or feet.shape != (n, 2, 3)
        or force.shape != (n,)
        or forbidden.shape != (n,)
        or safe.shape != (n,)
        or forbidden.dtype != np.bool_
        or safe.dtype != np.bool_
        or any(not np.isfinite(a).all() for a in (ball, velocity, feet, force))
        or np.any(force < 0)
        or type(contact_index) is not int
        or not 1 <= contact_index < n
    ):
        raise ValueError("aligned finite measured reception arrays required")
    start = contact_index
    end = int(np.searchsorted(time, time[start] + config.observation_sec - 1e-9))
    complete = end < n
    stop = min(end, n - 1)
    tail = int(
        np.searchsorted(time, time[start] + config.observation_sec - config.stable_tail_sec - 1e-9)
    )
    speed = np.linalg.norm(velocity, axis=1)
    distance = np.min(np.linalg.norm(feet - ball[:, None, :], axis=2), axis=1)
    if not np.isfinite(speed).all() or not np.isfinite(distance).all():
        raise ValueError("reception measurement exceeds the finite numeric envelope")
    physical_touch = bool(force[start] > 0)
    incoming = float(speed[start - 1])
    clean = not bool(np.any(forbidden[start : stop + 1]))
    body = bool(np.all(safe[start : stop + 1]))
    maximum_speed = float(np.max(speed[tail : stop + 1])) if complete else None
    maximum_distance = float(np.max(distance[tail : stop + 1])) if complete else None
    trapped = bool(
        complete
        and physical_touch
        and incoming >= config.minimum_incoming_speed_mps
        and clean
        and body
        and maximum_speed is not None
        and maximum_speed <= config.maximum_speed_mps
        and maximum_distance is not None
        and maximum_distance <= config.maximum_foot_distance_m
    )
    failure = (
        "NO_PHYSICAL_FOOT_TOUCH"
        if not physical_touch
        else "INCOMPLETE_OBSERVATION"
        if not complete
        else "LOW_INCOMING_SPEED"
        if incoming < config.minimum_incoming_speed_mps
        else "FORBIDDEN_CONTACT"
        if not clean
        else "BODY_UNSAFE"
        if not body
        else "BALL_NOT_CONTROLLED"
        if not trapped
        else None
    )
    return dict(
        thresholds=asdict(config),
        kind="STOP_CONTROL",
        physical_foot_touch=physical_touch,
        contact_index=start,
        confirmation_index=end if complete else None,
        contact_time_sec=float(time[start]),
        incoming_speed_mps=incoming,
        complete_observation=complete,
        clean_contact_window=clean,
        body_safe_window=body,
        tail_maximum_ball_speed_mps=maximum_speed,
        tail_maximum_foot_distance_m=maximum_distance,
        controlled_reception=trapped,
        failure=failure,
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
    )
