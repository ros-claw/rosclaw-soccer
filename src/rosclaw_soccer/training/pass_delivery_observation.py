"""Read an upstream delivery event before receiver contact, without simulation.

Interpolated telemetry is an interface observation, never proof of a completed
pass, legal contact, stable receiver, goal, or policy promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.training.directed_plane_crossing import first_directed_plane_crossing


@dataclass(frozen=True)
class PassDeliveryObservation:
    segment_index: int
    fraction: float
    sample_interval_sec: tuple[float, float]
    time_sec: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    signed_lateral_error_m: float


def observe_pass_delivery(
    *,
    time_sec: NDArray[np.floating],
    position_m: NDArray[np.floating],
    velocity_mps: NDArray[np.floating],
    plane_origin_xy_m: NDArray[np.floating],
    plane_forward_xy: NDArray[np.floating],
    receiver_contact_sec: float | None,
) -> PassDeliveryObservation | None:
    """Read the first directed crossing whose entire bracket precedes contact.

    All three telemetry arrays must share the caller's declared sample clock;
    callers must not label pre-step positions with post-step timestamps. The
    native integration may be finer than this telemetry: interpolation does
    not establish the exact substep of crossing. Velocity is interpolated from
    supplied telemetry, not differentiated from position. A bracket ending at
    contact is excluded conservatively. ``None`` explicitly means no eligible
    observed crossing, not successful delivery or absence of contact.
    """
    arrays = (time_sec, position_m, velocity_mps)
    for array in arrays:
        if (
            not isinstance(array, np.ndarray)
            or array.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or array.size > 3_000_000
            or not np.isfinite(array).all()
            or (np.abs(array) > 1e6).any()
        ):
            raise ValueError("bounded finite floating-point telemetry required")
    if (
        time_sec.ndim != 1
        or len(time_sec) < 2
        or position_m.shape != (len(time_sec), 3)
        or velocity_mps.shape != (len(time_sec), 3)
        or (time_sec < 0).any()
        or (np.diff(time_sec) <= 0).any()
    ):
        raise ValueError("aligned XYZ samples and strictly increasing nonnegative clock required")
    if receiver_contact_sec is not None and (
        isinstance(receiver_contact_sec, bool)
        or not isinstance(receiver_contact_sec, (int, float))
        or not math.isfinite(receiver_contact_sec)
        or not 0 <= receiver_contact_sec <= 1e6
    ):
        raise ValueError("finite bounded receiver contact time or explicit None required")
    # Use the existing directed geometry contract, including validation even
    # when contact happens before the first sample.
    crossing = first_directed_plane_crossing(
        position_m[:, None, :2], origin=plane_origin_xy_m, forward=plane_forward_xy
    )
    if not crossing.crossed[0]:
        return None
    index = int(crossing.segment_index[0])
    if receiver_contact_sec is not None and time_sec[index + 1] >= receiver_contact_sec:
        return None
    fraction = float(crossing.fraction[0])
    position = position_m[index].astype(np.float64) + fraction * (
        position_m[index + 1].astype(np.float64) - position_m[index].astype(np.float64)
    )
    velocity = velocity_mps[index].astype(np.float64) + fraction * (
        velocity_mps[index + 1].astype(np.float64) - velocity_mps[index].astype(np.float64)
    )
    return PassDeliveryObservation(
        segment_index=index,
        fraction=fraction,
        sample_interval_sec=(float(time_sec[index]), float(time_sec[index + 1])),
        time_sec=float(time_sec[index])
        + fraction * (float(time_sec[index + 1]) - float(time_sec[index])),
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        velocity_mps=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
        signed_lateral_error_m=float(crossing.signed_lateral_error[0]),
    )
