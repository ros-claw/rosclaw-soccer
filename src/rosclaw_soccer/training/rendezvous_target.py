"""Bounded constant-velocity rendezvous proposals from current observations.

This predictor neither moves bodies nor certifies reception. Its explicit
kinematic assumptions require subsequent physical and team-protocol checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RendezvousPrediction:
    target_xy: tuple[float, float]
    predicted_contact_xy: tuple[float, float]
    incoming_contact_delay_sec: float
    outgoing_flight_sec: float
    horizon_sec: float


def predict_ground_rendezvous(
    *,
    ball_xy: tuple[float, float],
    ball_velocity_xy: tuple[float, float],
    passer_xy: tuple[float, float],
    passer_velocity_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    receiver_velocity_xy: tuple[float, float],
    contact_standoff_m: float,
    outgoing_speed_mps: float,
    maximum_horizon_sec: float = 4.0,
) -> RendezvousPrediction:
    """Predict approaching contact, then intercept a constant-velocity receiver.

    Uses a radial closing-speed contact estimate and an exact planar outgoing
    interception solution. Assumes rolling motion, no obstacles, no acceleration
    and a receiver slower than the expected pass. These are not success claims.
    """
    vectors = (
        ball_xy,
        ball_velocity_xy,
        passer_xy,
        passer_velocity_xy,
        receiver_xy,
        receiver_velocity_xy,
    )
    if any(
        type(v) is not tuple
        or len(v) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000 for x in v)
        for v in vectors
    ) or any(
        type(x) not in (int, float) or not math.isfinite(x)
        for x in (contact_standoff_m, outgoing_speed_mps, maximum_horizon_sec)
    ):
        raise ValueError("finite bounded planar observations required")
    if not (
        0.01 <= contact_standoff_m <= 0.5
        and 0.2 <= outgoing_speed_mps <= 15
        and 0.1 <= maximum_horizon_sec <= 5
        and max(math.hypot(*v) for v in vectors[1::2]) <= 15
        and math.hypot(*receiver_velocity_xy) < outgoing_speed_mps
    ):
        raise ValueError("bounded contact, horizon and interceptable receiver required")
    relative = tuple(b - p for b, p in zip(ball_xy, passer_xy, strict=True))
    distance = math.hypot(*relative)
    closing = -sum(
        d * (b - p) for d, b, p in zip(relative, ball_velocity_xy, passer_velocity_xy, strict=True)
    ) / max(distance, 1e-9)
    if distance > contact_standoff_m and closing <= 1e-6:
        raise ValueError("incoming ball is not approaching the contact region")
    delay = max(0.0, distance - contact_standoff_m) / max(closing, 1e-6)
    release = tuple(b + v * delay for b, v in zip(ball_xy, ball_velocity_xy, strict=True))
    delta = tuple(
        p + v * delay - b
        for p, v, b in zip(receiver_xy, receiver_velocity_xy, release, strict=True)
    )
    squared_distance = sum(x * x for x in delta)
    if squared_distance < 1e-8:
        raise ValueError("distinct outgoing receiver location required")
    speed_gap = outgoing_speed_mps**2 - sum(v * v for v in receiver_velocity_xy)
    linear = 2 * sum(d * v for d, v in zip(delta, receiver_velocity_xy, strict=True))
    denominator = math.sqrt(linear * linear + 4 * speed_gap * squared_distance) - linear
    if denominator <= 0:
        raise ValueError("unresolved outgoing interception")
    flight = 2 * squared_distance / denominator
    horizon = delay + flight
    if not math.isfinite(horizon) or horizon > maximum_horizon_sec:
        raise ValueError("rendezvous exceeds prediction horizon")
    target = tuple(p + v * horizon for p, v in zip(receiver_xy, receiver_velocity_xy, strict=True))
    return RendezvousPrediction(
        (target[0], target[1]), (release[0], release[1]), delay, flight, horizon
    )
