"""Measured goal-plane terminal error cost, separate from potential shaping.

This is task feedback only, never goal adjudication, physical safety or policy
promotion. Callers bind the actual crossing and complete clean-episode evidence;
virtual conditioning targets and ballistic predictions are not valid inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ShotPrecisionRewardConfig:
    error_weight_per_m: float = 4.0
    missing_or_failed_error_m: float = 4.0

    def __post_init__(self) -> None:
        values = (self.error_weight_per_m, self.missing_or_failed_error_m)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 < self.error_weight_per_m <= 20
            or not 0.1 <= self.missing_or_failed_error_m <= 20
        ):
            raise ValueError("bounded finite shot-precision reward configuration required")


def terminal_shot_precision_penalty(
    *,
    measured_crossing_error_m: float | None,
    complete_clean_episode: bool,
    config: ShotPrecisionRewardConfig | None = None,
) -> float:
    """Add a terminal objective that cannot telescope away as shaping does.

    Missing crossings and incomplete/unclean episodes receive the maximum
    bounded cost, even if a ball previously crossed near the target. Distances
    above the declared cap also saturate. This changes the learning objective
    and requires a new protocol and fresh collection, not edited old evidence.
    """
    active = ShotPrecisionRewardConfig() if config is None else config
    if (
        not isinstance(active, ShotPrecisionRewardConfig)
        or type(complete_clean_episode) is not bool
    ):
        raise ValueError("validated precision config and explicit clean-episode flag required")
    if measured_crossing_error_m is not None and (
        type(measured_crossing_error_m) not in (int, float)
        or not math.isfinite(measured_crossing_error_m)
        or measured_crossing_error_m < 0
    ):
        raise ValueError("measured crossing error must be finite and nonnegative, or absent")
    error = (
        min(measured_crossing_error_m, active.missing_or_failed_error_m)
        if complete_clean_episode and measured_crossing_error_m is not None
        else active.missing_or_failed_error_m
    )
    return float(-active.error_weight_per_m * error)
