"""SIM_ONLY numerical approach proposals; never an actuator or learned policy.

Parameters describe a development hypothesis. Native skill/retention evidence
is still required before using a proposal in a candidate controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RunningKickApproachConfig:
    approach_deceleration_m_s2: float = 1.0
    contact_speed_m_s: float = 0.5
    braking_gap_m: float = 1.0
    maximum_entry_gap_m: float = 0.8
    maximum_entry_speed_m_s: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.approach_deceleration_m_s2,
            self.contact_speed_m_s,
            self.braking_gap_m,
            self.maximum_entry_gap_m,
            self.maximum_entry_speed_m_s,
        )
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("finite numerical approach parameters required")
        if not (
            0.1 <= self.approach_deceleration_m_s2 <= 4.0
            and 0.1 <= self.contact_speed_m_s <= self.maximum_entry_speed_m_s <= 2.0
            and 0 < self.maximum_entry_gap_m <= self.braking_gap_m <= 4.0
        ):
            raise ValueError("bounded ordered approach parameters required")


@dataclass(frozen=True)
class RunningKickApproachProposal:
    requested_forward_speed_m_s: float
    entry_condition_met: bool
    activation_ceiling: str = "SIM_ONLY"
    promotion_eligible: bool = False


def propose_running_kick_approach(
    *,
    longitudinal_ball_gap_m: float,
    measured_forward_speed_m_s: float,
    cruising_speed_m_s: float,
    config: RunningKickApproachConfig | None = None,
) -> RunningKickApproachProposal:
    """Use current observations only; caller owns heading and mode transitions.

    Gap and speed must be expressed along the same intended approach axis.
    A behind-the-body ball requests zero approach speed and never satisfies
    entry; the caller must reorient or replan. This proposal neither
    certifies foot placement/balance nor guarantees actual nonzero motion.
    The caller must latch entry and stop calling this approach-only function
    when another controller owns the action. Invalid observations raise and
    must not be converted into a motion request by the caller.
    """
    active = RunningKickApproachConfig() if config is None else config
    if not isinstance(active, RunningKickApproachConfig):
        raise ValueError("validated approach config required")
    values = (longitudinal_ball_gap_m, measured_forward_speed_m_s, cruising_speed_m_s)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("finite scalar approach observations required")
    if not (
        abs(longitudinal_ball_gap_m) <= 30
        and abs(measured_forward_speed_m_s) <= 10
        and active.contact_speed_m_s <= cruising_speed_m_s <= 4
    ):
        raise ValueError("approach observations outside declared simulation envelope")
    requested = min(
        cruising_speed_m_s,
        math.sqrt(
            active.contact_speed_m_s**2
            + 2
            * active.approach_deceleration_m_s2
            * max(longitudinal_ball_gap_m - active.braking_gap_m, 0.0)
        ),
    )
    if longitudinal_ball_gap_m <= 0:
        requested = 0.0
    return RunningKickApproachProposal(
        requested_forward_speed_m_s=float(requested),
        entry_condition_met=bool(
            0 < longitudinal_ball_gap_m <= active.maximum_entry_gap_m
            and 0 < measured_forward_speed_m_s <= active.maximum_entry_speed_m_s
        ),
    )
