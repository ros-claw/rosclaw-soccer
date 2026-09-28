"""Causal, simulation-only pre-contact gate for moving-ball motor research."""

from __future__ import annotations

import math


def contact_adapter_window_open(
    *,
    ball_root_gap_m: float,
    ball_origin_displacement_m: float,
    first_contact_seen: bool,
    rolling_adapter_enabled: bool,
    maximum_gap_m: float,
) -> bool:
    """Use the current ball/body state, never an eventual collision label.

    The legacy static-ball path retains its old origin displacement check.
    An explicitly enabled rolling-ball path instead stops at the first measured
    body/ball contact. Neither path grants actuator or hardware authority.
    """
    if (
        not all(
            math.isfinite(value)
            for value in (ball_root_gap_m, ball_origin_displacement_m, maximum_gap_m)
        )
        or ball_origin_displacement_m < 0.0
        or maximum_gap_m not in (0.75, 1.0)
        or type(first_contact_seen) is not bool
        or type(rolling_adapter_enabled) is not bool
    ):
        raise ValueError("finite bounded pre-contact window inputs required")
    return bool(
        0.1 <= ball_root_gap_m <= maximum_gap_m
        and (
            not first_contact_seen if rolling_adapter_enabled else ball_origin_displacement_m < 0.05
        )
    )
