"""Causal lower bound for an observed SIM pass commitment and foot contact."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PassLaunchEvidence:
    commitment_visible_before_touch: bool
    velocity_toward_receiver_mps: float | None
    directed_launch_observed: bool
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False


def inspect_pass_launch(
    *,
    control_time_sec: np.ndarray,
    source_codes: np.ndarray,
    receiver_codes: np.ndarray,
    sender_code: int,
    receiver_code: int,
    contact_time_sec: float,
    ball_xy_m: tuple[float, float],
    ball_velocity_xy_mps: tuple[float, float],
    receiver_position_xy_m: tuple[float, float],
    minimum_directed_speed_mps: float = 0.4,
) -> PassLaunchEvidence:
    """Check the last completed control snapshot strictly before a foot event.

    Caller must separately authenticate the stream, its true sender-foot event,
    team identity and target, safety and any receipt. The receiver position is
    a local observed proxy, not necessarily the immutable tactical target;
    this is a conservative diagnostic, never proof of intention or causality.
    """
    t = np.asarray(control_time_sec)
    src, dst = np.asarray(source_codes), np.asarray(receiver_codes)
    if (
        t.ndim != 1
        or not len(t)
        or t.dtype.kind not in "fiu"
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or src.shape != t.shape
        or dst.shape != t.shape
        or src.dtype.kind not in "iu"
        or dst.dtype.kind not in "iu"
        or type(sender_code) is not int
        or type(receiver_code) is not int
        or sender_code <= 0
        or receiver_code <= 0
        or sender_code == receiver_code
        or type(contact_time_sec) not in (float, int)
        or not np.isfinite(contact_time_sec)
        or type(minimum_directed_speed_mps) is not float
        or not np.isfinite(minimum_directed_speed_mps)
        or minimum_directed_speed_mps <= 0
    ):
        raise ValueError(
            "finite ordered clock, distinct explicit agent codes and threshold required"
        )
    vectors = (ball_xy_m, ball_velocity_xy_mps, receiver_position_xy_m)
    if any(
        type(value) is not tuple
        or len(value) != 2
        or any(type(x) not in (int, float) or not np.isfinite(x) for x in value)
        for value in vectors
    ):
        raise ValueError("finite measured planar vectors required")
    prior = int(np.searchsorted(t, contact_time_sec, side="left")) - 1
    if prior < 0 or contact_time_sec - float(t[prior]) > 0.020001:
        visible = False
    else:
        visible = bool(src[prior] == sender_code and dst[prior] == receiver_code)
    direction = np.asarray(receiver_position_xy_m) - np.asarray(ball_xy_m)
    distance = float(np.linalg.norm(direction))
    velocity = (
        None
        if distance <= 1e-9
        else float(np.dot(np.asarray(ball_velocity_xy_mps), direction) / distance)
    )
    directed = velocity is not None and velocity >= minimum_directed_speed_mps
    reasons = []
    if not visible:
        reasons.append("no_matching_precontact_commitment")
    if velocity is None:
        reasons.append("receiver_direction_undefined")
    elif not directed:
        reasons.append("insufficient_velocity_toward_receiver")
    return PassLaunchEvidence(visible, velocity, bool(visible and directed), tuple(reasons))
