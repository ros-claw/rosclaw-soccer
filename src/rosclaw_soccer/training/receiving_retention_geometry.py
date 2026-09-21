"""Post-touch prediction filter, not measured success or execution authority.

Acquisition needs a foot collision; retaining a ball already touched does not
need another collision in every prediction horizon. The caller owns the measured
contact event and must reset it on a new episode/possession. This pure helper
cannot authenticate that provenance, qualify a model, or replace the real exam.
"""

import math
from collections.abc import Mapping
from typing import Any

from rosclaw_soccer.training.receiving_bias_search import ReceivingBiasForecast


def receiving_departure_margin(
    *, distance_m: float, radial_departure_mps: float, projection_sec: float = 0.5
) -> float:
    """Linear pocket margin using ball-minus-foot velocity projected outwards.

    Positive means only that the linear extrapolation stays within 0.35 m.
    It is NOT a control barrier certificate: acceleration, foot switching,
    contacts and policy changes are not modeled. Never use it as the real exam.
    Closing velocity does not create extra credit beyond the current margin.
    Geometry and velocity must refer to the same instant and selected foot.
    """
    for value in (distance_m, radial_departure_mps, projection_sec):
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e4:
            raise ValueError("finite bounded real geometry, velocity and projection required")
    if distance_m < 0 or not 0 < projection_sec <= 2:
        raise ValueError("nonnegative distance and projection in (0, 2] seconds required")
    return float(0.35 - distance_m - projection_sec * max(radial_departure_mps, 0.0))


def receiving_retention_feasible(
    diagnostics: Mapping[str, Any],
    *,
    prior_own_touch_sec: float | None,
    observation_time_sec: float,
) -> bool:
    """Fixed research filter with explicit past measured-contact precondition.

    Distance/speed/height are predicted terminal values, not duration-in-control
    or successor readiness. A true result grants no motor/promotion permission.
    Malformed inputs fail loudly rather than disappearing from search evidence.
    """
    forecast = ReceivingBiasForecast.from_mapping(diagnostics)
    if (
        type(observation_time_sec) not in (int, float)
        or not math.isfinite(observation_time_sec)
        or observation_time_sec < 0
    ):
        raise ValueError("finite nonnegative observation time required")
    if prior_own_touch_sec is not None and (
        type(prior_own_touch_sec) not in (int, float)
        or not math.isfinite(prior_own_touch_sec)
        or not 0 <= prior_own_touch_sec <= observation_time_sec + 1e-9
    ):
        # Match ReceivingFeedbackObservation's simulator/50Hz clock roundoff;
        # this is not a prediction horizon or physical threshold relaxation.
        raise ValueError("measured own-foot contact must not be in the future")
    try:
        distance = diagnostics["terminal_distance_m"]
        height = diagnostics["terminal_height_m"]
    except KeyError as error:
        raise ValueError("explicit predicted terminal distance and height required") from error
    for value in (distance, height):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e4:
            raise ValueError("finite nonnegative bounded predicted geometry required")
    return (
        prior_own_touch_sec is not None
        and not forecast.unsafe
        and forecast.nonfoot_contact_samples == 0
        and forecast.terminal_ball_speed <= 0.35
        and distance <= 0.35
        and height <= 0.2
    )
